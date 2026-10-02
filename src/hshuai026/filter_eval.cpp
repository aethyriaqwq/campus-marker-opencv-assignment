// 时序滤波方案对比：在真实检测轨迹上评测 EMA / α-β / KF-CV / KF-CA /
// One-Euro / KF-CV+延迟补偿 的相位滞后、抖动与保真度。
//
// 用法: filter_eval <pose_or_log.csv> [输出csv]
//   CSV 需要包含 frame,found,cx,cy 四列（marker_video_log.csv 即满足）。
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Point {
    double x = 0.0;
    double y = 0.0;
};

struct Segment {
    std::vector<int> frames;
    std::vector<Point> raw;
};

std::vector<Segment> loadSegments(const std::string& path) {
    std::ifstream in(path);
    std::vector<Segment> segments;
    Segment current;
    std::string line;
    std::getline(in, line);  // 表头
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream stream(line);
        std::string field;
        std::vector<std::string> fields;
        while (std::getline(stream, field, ',')) {
            fields.push_back(field);
        }
        if (fields.size() < 4) {
            continue;
        }
        const int frame = std::atoi(fields[0].c_str());
        const int found = std::atoi(fields[1].c_str());
        if (found == 0) {
            if (current.raw.size() > 5) {
                segments.push_back(current);
            }
            current = Segment{};
            continue;
        }
        current.frames.push_back(frame);
        current.raw.push_back({std::atof(fields[2].c_str()),
                               std::atof(fields[3].c_str())});
    }
    if (current.raw.size() > 5) {
        segments.push_back(current);
    }
    return segments;
}

double mean(const std::vector<double>& values) {
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
}

// 二阶差分（加速度）的标准差，作为"抖动"指标。
double jitter(const std::vector<Point>& track, int from, int to) {
    std::vector<double> values;
    for (int i = from + 1; i < to - 1; ++i) {
        const double ax = track[i + 1].x - 2.0 * track[i].x + track[i - 1].x;
        const double ay = track[i + 1].y - 2.0 * track[i].y + track[i - 1].y;
        values.push_back(std::sqrt(ax * ax + ay * ay));
    }
    if (values.size() < 2) {
        return 0.0;
    }
    const double m = mean(values);
    double sum = 0.0;
    for (double value : values) {
        sum += (value - m) * (value - m);
    }
    return std::sqrt(sum / static_cast<double>(values.size()));
}

// 用互相关求滞后：filtered[t] ≈ raw[t - lag] 时返回正的 lag（帧）。
double estimateLag(const std::vector<Point>& raw, const std::vector<Point>& filtered,
                   int max_shift) {
    const int n = static_cast<int>(raw.size());
    if (n < 2 * max_shift + 8) {
        return 0.0;
    }
    double best_shift = 0.0;
    double best_corr = -2.0;
    for (int shift = -max_shift; shift <= max_shift; ++shift) {
        std::vector<double> a;
        std::vector<double> b;
        for (int t = max_shift; t < n - max_shift; ++t) {
            const int source = t - shift;  // filtered[t] 对应 raw[t - lag]
            if (source < 0 || source >= n) {
                continue;
            }
            a.push_back(raw[source].x);
            b.push_back(filtered[t].x);
            a.push_back(raw[source].y);
            b.push_back(filtered[t].y);
        }
        if (a.size() < 16) {
            continue;
        }
        const double ma = mean(a);
        const double mb = mean(b);
        double numerator = 0.0;
        double da = 0.0;
        double db = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            numerator += (a[i] - ma) * (b[i] - mb);
            da += (a[i] - ma) * (a[i] - ma);
            db += (b[i] - mb) * (b[i] - mb);
        }
        const double corr = numerator / std::sqrt(da * db + 1e-12);
        if (corr > best_corr) {
            best_corr = corr;
            best_shift = shift;
        }
    }
    return best_shift;
}

// ---------------- 滤波器 ----------------
std::vector<Point> ema(const std::vector<Point>& input, double alpha) {
    std::vector<Point> out;
    Point state = input.front();
    for (const Point& p : input) {
        state.x = alpha * p.x + (1 - alpha) * state.x;
        state.y = alpha * p.y + (1 - alpha) * state.y;
        out.push_back(state);
    }
    return out;
}

std::vector<Point> alphaBeta(const std::vector<Point>& input, double alpha,
                             double beta) {
    std::vector<Point> out;
    Point state = input.front();
    Point velocity{0.0, 0.0};
    for (const Point& p : input) {
        const Point predicted{state.x + velocity.x, state.y + velocity.y};
        const Point residual{p.x - predicted.x, p.y - predicted.y};
        state.x = predicted.x + alpha * residual.x;
        state.y = predicted.y + alpha * residual.y;
        velocity.x += beta * residual.x;
        velocity.y += beta * residual.y;
        out.push_back(predicted);  // 输出预测值，减少相位滞后
    }
    return out;
}

// 匀速卡尔曼滤波（状态 [x, y, vx, vy]），predict_steps 为输出前额外预测的步数。
std::vector<Point> kalmanCV(const std::vector<Point>& input, double q, double r,
                            int predict_steps = 0) {
    // 状态向量与协方差（手工实现，维度固定为 4）。
    double x[4] = {input.front().x, input.front().y, 0.0, 0.0};
    double P[4][4] = {};
    for (int i = 0; i < 4; ++i) {
        P[i][i] = (i < 2) ? 100.0 : 1000.0;
    }
    const double F[4][4] = {{1, 0, 1, 0}, {0, 1, 0, 1}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    std::vector<Point> out;
    for (const Point& p : input) {
        // 预测
        double xp[4] = {};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                xp[i] += F[i][j] * x[j];
            }
        }
        double Pp[4][4] = {};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 4; ++k) {
                    for (int l = 0; l < 4; ++l) {
                        sum += F[i][k] * P[k][l] * F[j][l];
                    }
                }
                Pp[i][j] = sum + ((i == j) ? q : 0.0);
            }
        }
        // 更新（观测为位置）
        const double H[2][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}};
        const double z[2] = {p.x, p.y};
        double y[2];
        for (int i = 0; i < 2; ++i) {
            y[i] = z[i];
            for (int j = 0; j < 4; ++j) {
                y[i] -= H[i][j] * xp[j];
            }
        }
        double PHt[4][2] = {};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 4; ++k) {
                    PHt[i][j] += Pp[i][k] * H[j][k];
                }
            }
        }
        double S[2][2] = {};
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 4; ++k) {
                    sum += H[i][k] * PHt[k][j];
                }
                S[i][j] = sum + ((i == j) ? r : 0.0);
            }
        }
        const double det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
        double Sinv[2][2] = {{S[1][1] / det, -S[0][1] / det},
                             {-S[1][0] / det, S[0][0] / det}};
        double K[4][2] = {};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 2; ++k) {
                    K[i][j] += PHt[i][k] * Sinv[k][j];
                }
            }
        }
        double correction[4] = {};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 2; ++j) {
                correction[i] += K[i][j] * y[j];
            }
        }
        for (int i = 0; i < 4; ++i) {
            x[i] = xp[i] + correction[i];
        }
        // P = (I - K H) * Pp
        double KH[4][4] = {};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 2; ++k) {
                    KH[i][j] += K[i][k] * H[k][j];
                }
            }
        }
        double Pn[4][4] = {};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 4; ++k) {
                    sum += KH[i][k] * Pp[k][j];
                }
                Pn[i][j] = Pp[i][j] - sum;
            }
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                P[i][j] = 0.5 * (Pn[i][j] + Pn[j][i]);  // 保持对称
            }
        }
        // 输出：再向前预测 predict_steps 步（延迟补偿）
        out.push_back({x[0] + predict_steps * x[2], x[1] + predict_steps * x[3]});
    }
    return out;
}

// 一欧元滤波：截止频率随速度自适应（低速重平滑、高速低滞后）。
std::vector<Point> oneEuro(const std::vector<Point>& input, double min_cutoff,
                           double beta, double d_cutoff) {
    std::vector<Point> out;
    Point filtered = input.front();
    Point derivative{0.0, 0.0};
    double last_x = input.front().x;
    double last_y = input.front().y;
    auto alpha = [](double cutoff) {
        const double te = 1.0;  // 采样周期归一化为 1 帧
        const double tau = 1.0 / (2.0 * M_PI * cutoff);
        return 1.0 / (1.0 + tau / te);
    };
    for (const Point& p : input) {
        const Point d{p.x - last_x, p.y - last_y};
        const double ad = alpha(d_cutoff);
        derivative.x = ad * d.x + (1 - ad) * derivative.x;
        derivative.y = ad * d.y + (1 - ad) * derivative.y;
        last_x = p.x;
        last_y = p.y;
        const double speed = std::sqrt(derivative.x * derivative.x +
                                       derivative.y * derivative.y);
        const double cutoff = min_cutoff + beta * speed;
        const double a = alpha(cutoff);
        filtered.x = a * p.x + (1 - a) * filtered.x;
        filtered.y = a * p.y + (1 - a) * filtered.y;
        out.push_back(filtered);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "用法: filter_eval <log.csv> [输出csv]\n";
        return 1;
    }
    const std::vector<Segment> segments = loadSegments(argv[1]);
    if (segments.empty()) {
        std::cerr << "没有读到有效轨迹（需要 frame,found,cx,cy 列）\n";
        return 1;
    }
    std::size_t total = 0;
    for (const Segment& segment : segments) {
        total += segment.raw.size();
    }
    std::cout << "读取 " << segments.size() << " 段轨迹，共 " << total << " 个检测点\n";

    struct Candidate {
        std::string name;
        std::vector<Point> (*run)(const std::vector<Point>&);
    };
    const std::vector<std::pair<std::string, std::vector<Point>>> results = [] {
        std::vector<std::pair<std::string, std::vector<Point>>> list;
        return list;
    }();
    (void)results;

    std::cout << std::left << std::setw(22) << "方案" << std::setw(12) << "滞后(帧)"
              << std::setw(14) << "抖动(px)" << std::setw(14) << "与raw的RMSE(px)"
              << "\n";

    std::vector<std::string> names = {"EMA(a=0.35) 现状", "α-β(a=0.5,b=0.2)",
                                      "KF-CV(q=0.5,r=9)", "KF-CV+延迟补偿1帧",
                                      "One-Euro", "KF-CV 强平滑"};
    std::vector<std::vector<Point>> tables;
    for (const auto& name : names) {
        tables.emplace_back();
    }

    for (const Segment& segment : segments) {
        const auto raw = segment.raw;
        std::vector<std::vector<Point>> outputs;
        outputs.push_back(ema(raw, 0.35));
        outputs.push_back(alphaBeta(raw, 0.5, 0.2));
        outputs.push_back(kalmanCV(raw, 0.5, 9.0, 0));
        outputs.push_back(kalmanCV(raw, 0.5, 9.0, 1));
        outputs.push_back(oneEuro(raw, 0.6, 0.02, 1.0));
        outputs.push_back(kalmanCV(raw, 0.1, 25.0, 0));
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            tables[i].insert(tables[i].end(), outputs[i].begin(), outputs[i].end());
        }
        if (tables[0].size() != tables[1].size()) {
            std::cerr << "输出长度不一致\n";
            return 1;
        }
    }

    // 汇总统计（所有段拼接后统一计算）。
    std::vector<Point> raw_all;
    for (const Segment& segment : segments) {
        raw_all.insert(raw_all.end(), segment.raw.begin(), segment.raw.end());
    }
    std::vector<double> baseline_jitter;
    std::vector<double> baseline_rmse;
    for (std::size_t i = 1; i + 1 < raw_all.size(); ++i) {
        const double ax = raw_all[i + 1].x - 2 * raw_all[i].x + raw_all[i - 1].x;
        const double ay = raw_all[i + 1].y - 2 * raw_all[i].y + raw_all[i - 1].y;
        baseline_jitter.push_back(std::sqrt(ax * ax + ay * ay));
    }
    const double jitter_raw = [&] {
        const double m = mean(baseline_jitter);
        double sum = 0.0;
        for (double value : baseline_jitter) {
            sum += (value - m) * (value - m);
        }
        return std::sqrt(sum / std::max<std::size_t>(1, baseline_jitter.size()));
    }();
    std::cout << "原始检测抖动基准: " << std::round(jitter_raw * 100.0) / 100.0
              << " px\n\n";

    int index = 0;
    for (std::size_t i = 0; i < names.size(); ++i) {
        const auto& filtered = tables[i];
        if (filtered.size() != raw_all.size()) {
            continue;
        }
        // 逐段求滞后，再取平均。
        double lag_sum = 0.0;
        int lag_count = 0;
        std::size_t offset = 0;
        for (const Segment& segment : segments) {
            std::vector<Point> raw_seg(segment.raw);
            std::vector<Point> out_seg(filtered.begin() + offset,
                                       filtered.begin() + offset + raw_seg.size());
            lag_sum += estimateLag(raw_seg, out_seg, 6);
            ++lag_count;
            offset += raw_seg.size();
        }
        std::vector<double> values;
        for (std::size_t k = 1; k + 1 < filtered.size(); ++k) {
            const double ax = filtered[k + 1].x - 2 * filtered[k].x + filtered[k - 1].x;
            const double ay = filtered[k + 1].y - 2 * filtered[k].y + filtered[k - 1].y;
            values.push_back(std::sqrt(ax * ax + ay * ay));
        }
        const double m = mean(values);
        double sum = 0.0;
        for (double value : values) {
            sum += (value - m) * (value - m);
        }
        const double jitter_filtered =
            std::sqrt(sum / std::max<std::size_t>(1, values.size()));
        double rmse = 0.0;
        for (std::size_t k = 0; k < filtered.size(); ++k) {
            const double dx = filtered[k].x - raw_all[k].x;
            const double dy = filtered[k].y - raw_all[k].y;
            rmse += dx * dx + dy * dy;
        }
        rmse = std::sqrt(rmse / static_cast<double>(filtered.size()));
        std::cout << std::left << std::setw(22) << names[i] << std::setw(12)
                  << std::fixed << std::setprecision(2) << (lag_sum / lag_count)
                  << std::setw(14) << jitter_filtered << std::setw(14) << rmse
                  << "\n";
        ++index;
    }

    if (argc > 2) {
        std::ofstream out(argv[2]);
        out << "frame,raw_x,raw_y,ema_x,ema_y,ab_x,ab_y,kf_x,kf_y,kf1_x,kf1_y,"
               "oneeuro_x,oneeuro_y,kfs_x,kfs_y\n";
        std::size_t k = 0;
        for (const Segment& segment : segments) {
            for (std::size_t i = 0; i < segment.raw.size(); ++i, ++k) {
                out << segment.frames[i] << ',' << segment.raw[i].x << ','
                    << segment.raw[i].y << ',' << tables[0][k].x << ',' << tables[0][k].y
                    << ',' << tables[1][k].x << ',' << tables[1][k].y << ','
                    << tables[2][k].x << ',' << tables[2][k].y << ',' << tables[3][k].x
                    << ',' << tables[3][k].y << ',' << tables[4][k].x << ','
                    << tables[4][k].y << ',' << tables[5][k].x << ',' << tables[5][k].y
                    << '\n';
            }
        }
        std::cout << "\n轨迹对比已输出: " << argv[2] << "\n";
    }
    return 0;
}
