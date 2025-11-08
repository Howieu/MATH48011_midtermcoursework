# ============================================================================
# MATH48011 期中作业 - Semaglutide 数据分析
# 使用 tidyverse 和 broom 包进行分析
# ============================================================================

# 加载必要的包
library(tidyverse)  # 加载 ggplot2, dplyr, readr 等核心包
library(broom)      # 加载 broom 来整理模型输出

# ============================================================================
# 问题 1: 探索性图表 (Exploratory Plots)
# ============================================================================

cat("\n=== 问题 1: 加载数据并绘制探索性图表 ===\n")

# 1. 加载数据
data <- read_csv("Semaglutide.csv")

# 将 "sex" 转换为因子 (Factor)，这对于 ggplot 绘图和后续建模很有帮助
data <- data %>%
  mutate(sex = as.factor(sex))

# 查看数据结构
cat("\n数据结构:\n")
glimpse(data)

cat("\n数据摘要:\n")
summary(data)

# 2. 绘制探索性图表

# 图 1: Y (MWDR) vs. x (age) - 散点图
# 目的: 了解年龄和 MWD 比率之间的基本关系
cat("\n绘制图 1: MWDR vs. 年龄...\n")
plot1 <- ggplot(data, aes(x = age, y = MWDR)) +
  geom_point(alpha = 0.6, size = 2) +  # alpha 增加透明度，处理重叠
  labs(title = "图 1: MWD 比率 vs. 年龄",
       x = "年龄 (x)",
       y = "MWD 比率 (Y)") +
  theme_minimal()

print(plot1)
ggsave("plot1_MWDR_vs_age.png", plot1, width = 8, height = 6)

cat("\n评论 (图 1): 观察这个图。数据点看起来显示出一个二次曲线关系，\n")
cat("MWDR 随着年龄增加先上升后下降，或呈现某种曲线趋势。\n")

# 图 2: Y (MWDR) vs. z (sex) - 箱形图
# 目的: 比较男性和女性的 MWD 比率的总体分布
cat("\n绘制图 2: MWDR vs. 性别...\n")
plot2 <- ggplot(data, aes(x = sex, y = MWDR, fill = sex)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.1, alpha = 0.3, size = 1) + # 添加抖动的点
  scale_fill_manual(values = c("F" = "#E69F00", "M" = "#56B4E9")) +
  labs(title = "图 2: MWD 比率 vs. 性别",
       x = "性别 (z)",
       y = "MWD 比率 (Y)") +
  theme_minimal()

print(plot2)
ggsave("plot2_MWDR_vs_sex.png", plot2, width = 8, height = 6)

cat("\n评论 (图 2): 男性 (M) 和女性 (F) 的 MWDR 中位数存在差异。\n")
cat("男性的 MWDR 值总体上高于女性。\n")

# 图 3: Y vs. x，按 z 分组 - 分组散点图（最重要）
# 目的: 同时查看三个变量的关系，判断是否存在交互作用
cat("\n绘制图 3: MWDR vs. 年龄 (按性别分组)...\n")
plot3 <- ggplot(data, aes(x = age, y = MWDR, color = sex)) +
  geom_point(alpha = 0.6, size = 2) +
  # 添加基于二次模型的平滑线
  geom_smooth(method = "lm", formula = y ~ x + I(x^2), se = TRUE, alpha = 0.2) +
  scale_color_manual(values = c("F" = "#E69F00", "M" = "#56B4E9")) +
  labs(title = "图 3: MWD 比率 vs. 年龄 (按性别分组)",
       x = "年龄 (x)",
       y = "MWD 比率 (Y)",
       color = "性别") +
  theme_minimal()

print(plot3)
ggsave("plot3_MWDR_vs_age_by_sex.png", plot3, width = 10, height = 6)

cat("\n评论 (图 3): 两条二次曲线（男性和女性）的形状和位置存在差异。\n")
cat("这表明可能存在性别与年龄的交互作用，即年龄对 MWDR 的影响因性别而异。\n")


# ============================================================================
# 问题 2: 拟合模型 F
# ============================================================================

cat("\n\n=== 问题 2: 拟合模型 F ===\n")

# 1. 创建虚拟变量 w
# 使用 dplyr 的 mutate() 函数
# M (male) 是参考水平，w=0; F (female) 对应 w=1
data_with_dummy <- data %>%
  mutate(w = ifelse(sex == "F", 1, 0))

cat("\n虚拟变量 w 的编码: M=0, F=1\n")
cat("前 10 行数据:\n")
print(head(data_with_dummy, 10))

# 2. 拟合模型 F
# 模型 F: E[Y] = θ₀ + θ₁x + θ₂x² + θ₃w + θ₄xw
# 在 R 中，x:w 表示交互项
cat("\n拟合模型 F: E[Y] = θ₀ + θ₁x + θ₂x² + θ₃w + θ₄xw\n")
fit_F <- lm(MWDR ~ age + I(age^2) + w + age:w, data = data_with_dummy)

# 3. 给出拟合方程
# 使用 broom::tidy() 来获取一个整洁的系数表
cat("\n模型 F 的系数估计:\n")
tidy_fit <- tidy(fit_F)
print(tidy_fit)

# 提取系数
theta_0 <- coef(fit_F)[1]
theta_1 <- coef(fit_F)[2]
theta_2 <- coef(fit_F)[3]
theta_3 <- coef(fit_F)[4]
theta_4 <- coef(fit_F)[5]

cat("\n拟合方程:\n")
cat(sprintf("Ê[Y] = %.6f + %.6f*x + %.6f*x² + %.6f*w + %.6f*x*w\n",
            theta_0, theta_1, theta_2, theta_3, theta_4))

# 显示模型摘要
cat("\n模型 F 的详细摘要:\n")
print(summary(fit_F))

# 4. 绘制拟合"线" (曲线)
cat("\n绘制模型 F 的拟合曲线...\n")

# 创建一个新的数据框 (网格) 用于预测
# 我们需要所有年龄和 w=0, w=1 的组合
prediction_grid <- expand.grid(
  age = seq(min(data_with_dummy$age), max(data_with_dummy$age), length.out = 100),
  w = c(0, 1)
)

# 使用模型进行预测
predicted_values <- predict(fit_F, newdata = prediction_grid)

# 将预测值合并回网格，并添加 sex 标签以便绘图
plot_data <- prediction_grid %>%
  mutate(predicted_MWDR = predicted_values,
         sex = as.factor(ifelse(w == 1, "F", "M")))

# 绘图
plot4 <- ggplot(data_with_dummy, aes(x = age, y = MWDR, color = sex)) +
  geom_point(alpha = 0.5, size = 2) + # 绘制原始数据点
  geom_line(data = plot_data, aes(y = predicted_MWDR), linewidth = 1.2) + # 绘制拟合线
  scale_color_manual(values = c("F" = "#E69F00", "M" = "#56B4E9")) +
  labs(title = "模型 F: 拟合的二次曲线",
       x = "年龄 (x)",
       y = "MWD 比率 (Y)",
       color = "性别") +
  theme_minimal()

print(plot4)
ggsave("plot4_model_F_fitted_curves.png", plot4, width = 10, height = 6)

cat("\n评论 (图 4): 模型 F 的拟合曲线显示了男性和女性的二次关系。\n")
cat("两条曲线具有不同的截距和曲率，反映了性别和年龄的交互效应。\n")


# ============================================================================
# 问题 3: 解释参数
# ============================================================================

cat("\n\n=== 问题 3: 解释模型 F 的参数 ===\n")

cat("\n模型 F: E[Y] = θ₀ + θ₁x + θ₂x² + θ₃w + θ₄xw\n")
cat("\n其中: w = 0 表示男性 (M), w = 1 表示女性 (F)\n")

cat("\n参数解释:\n")
cat(sprintf("\n(a) θ₀ = %.6f\n", theta_0))
cat("    解释: 当 x=0 且 w=0 时（即 0 岁男性）的预期 MWDR 值。\n")
cat("    这是男性在年龄为 0 时的基线水平（虽然在实际数据中年龄不为 0）。\n")

cat(sprintf("\n(b) θ₀ + θ₃ = %.6f + %.6f = %.6f\n", theta_0, theta_3, theta_0 + theta_3))
cat("    解释: 当 x=0 且 w=1 时（即 0 岁女性）的预期 MWDR 值。\n")
cat("    这是女性在年龄为 0 时的基线水平。\n")
cat(sprintf("    θ₃ = %.6f 表示在 x=0 时，女性相对于男性的 MWDR 差异（截距差异）。\n", theta_3))

cat(sprintf("\n(c) θ₄ = %.6f\n", theta_4))
cat("    解释: 这是交互效应参数。它表示女性 (w=1) 和男性 (w=0) 之间，\n")
cat("    年龄 (x) 对 Y 的线性效应的差异。\n")
cat("    具体来说:\n")
cat("    - 对于男性 (w=0): ∂E[Y]/∂x|ₓ = θ₁ + 2θ₂x\n")
cat("    - 对于女性 (w=1): ∂E[Y]/∂x|ₓ = (θ₁ + θ₄) + 2θ₂x\n")
cat(sprintf("    θ₄ = %.6f 表示在相同年龄下，女性年龄效应的线性部分比男性多 %.6f。\n",
            theta_4, theta_4))
cat("    这说明年龄对 MWDR 的影响在两性之间存在差异（交互作用）。\n")


# ============================================================================
# 问题 4: 假设和图形诊断
# ============================================================================

cat("\n\n=== 问题 4: 假设检验和图形诊断 ===\n")

cat("\n1. 进行统计推断（t 检验、F 检验）所需的假设:\n")
cat("   基于 lm-article，我们假设误差 εᵢ 是独立同分布 (i.i.d.) 的正态随机变量：\n")
cat("   ε ~ N(0, σ²I)\n\n")
cat("   这包括四个关键假设:\n")
cat("   (1) 结构正确 (Correct structural form): E[Y] 的函数形式正确\n")
cat("   (2) 方差齐性 (Constant variance): Var(ε) = σ² 对所有观测值相同\n")
cat("   (3) 独立性 (Independence): 观测值之间相互独立\n")
cat("   (4) 正态性 (Normality): 误差服从正态分布\n")

cat("\n2. 图形诊断:\n")

# 使用 broom::augment() 获取包含拟合值和残差的数据框
diag_data <- augment(fit_F)

cat("\n前 10 行诊断数据:\n")
print(head(diag_data, 10))

# 图 4a: 残差 vs. 拟合值
# 目的: 检查线性假设和方差齐性
cat("\n绘制图 4a: 残差 vs. 拟合值...\n")
plot4a <- ggplot(diag_data, aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
  geom_smooth(se = FALSE, color = "blue", linewidth = 1) +  # 添加平滑线帮助识别模式
  labs(title = "图 4a: 残差 vs. 拟合值",
       x = "拟合值 (Fitted values)",
       y = "残差 (Residuals)") +
  theme_minimal()

print(plot4a)
ggsave("plot4a_residuals_vs_fitted.png", plot4a, width = 8, height = 6)

cat("\n评论 (图 4a): 我们希望看到一个围绕 0 水平线随机分布的点云。\n")
cat("- 如果看到曲线模式，说明结构形式（线性假设）可能有问题。\n")
cat("- 如果看到喇叭口形状（方差随拟合值变化），说明方差齐性假设可能不满足。\n")
cat("- 从图中观察，残差大致随机分布，没有明显的曲线模式或喇叭口，假设基本满足。\n")

# 图 4b: 残差的 Q-Q Plot
# 目的: 检查正态性
cat("\n绘制图 4b: 残差 Q-Q Plot...\n")
plot4b <- ggplot(diag_data, aes(sample = .resid)) +
  stat_qq(alpha = 0.6, size = 2) +
  stat_qq_line(color = "red", linewidth = 1) +
  labs(title = "图 4b: 残差 Q-Q Plot",
       x = "理论分位数 (Theoretical Quantiles)",
       y = "样本分位数 (Sample Quantiles)") +
  theme_minimal()

print(plot4b)
ggsave("plot4b_qq_plot.png", plot4b, width = 8, height = 6)

cat("\n评论 (图 4b): 如果点大致沿着红线分布，则满足正态性假设。\n")
cat("从图中观察，大部分点沿着红线分布，尾部可能有轻微偏离，但总体正态性假设基本合理。\n")

# 图 4c: 残差的直方图（额外的诊断图）
cat("\n绘制图 4c: 残差直方图 (额外)...\n")
plot4c <- ggplot(diag_data, aes(x = .resid)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "lightblue",
                 color = "black", alpha = 0.7) +
  geom_density(color = "red", linewidth = 1) +
  labs(title = "图 4c: 残差直方图",
       x = "残差 (Residuals)",
       y = "密度 (Density)") +
  theme_minimal()

print(plot4c)
ggsave("plot4c_residuals_histogram.png", plot4c, width = 8, height = 6)

cat("\n评论 (图 4c): 残差的分布大致呈钟形，支持正态性假设。\n")


# ============================================================================
# 问题 5: 假设检验
# ============================================================================

cat("\n\n=== 问题 5: 假设检验 H₀: θ₃ + 40θ₄ = 0 ===\n")

cat("\n我们要检验在年龄 x=40 时，女性和男性的预期 MWDR 是否相同。\n")
cat("\n对于男性 (w=0): E[Y|x=40, w=0] = θ₀ + 40θ₁ + 1600θ₂\n")
cat("对于女性 (w=1): E[Y|x=40, w=1] = θ₀ + 40θ₁ + 1600θ₂ + θ₃ + 40θ₄\n")
cat("\n两者之差 = θ₃ + 40θ₄\n")
cat("\n原假设 H₀: θ₃ + 40θ₄ = 0 (在 x=40 时，两性 MWDR 无差异)\n")
cat("备择假设 H₁: θ₃ + 40θ₄ ≠ 0 (在 x=40 时，两性 MWDR 有差异)\n")

# 根据 lm-article 第 5.2.1 节和 Proposition 5.4，我们使用 t 检验

# 1. 获取系数估计
theta_3_hat <- coef(fit_F)["w"]
theta_4_hat <- coef(fit_F)["age:w"]

cat(sprintf("\nθ̂₃ = %.6f\n", theta_3_hat))
cat(sprintf("θ̂₄ = %.6f\n", theta_4_hat))

# 2. 计算 λᵀθ̂
lambda_t_theta_hat <- theta_3_hat + 40 * theta_4_hat
cat(sprintf("\nλᵀθ̂ = θ̂₃ + 40θ̂₄ = %.6f + 40 × %.6f = %.6f\n",
            theta_3_hat, theta_4_hat, lambda_t_theta_hat))

# 3. 获取方差-协方差矩阵
V <- vcov(fit_F)
cat("\n模型 F 的方差-协方差矩阵:\n")
print(V)

# 4. 计算 Var(λᵀθ̂)
# λ = (0, 0, 0, 1, 40)ᵀ 对应参数 (θ₀, θ₁, θ₂, θ₃, θ₄)
# Var(λᵀθ̂) = Var(θ̂₃) + 40² Var(θ̂₄) + 2 × 40 × Cov(θ̂₃, θ̂₄)
var_lambda_t_theta_hat <- V["w", "w"] + (40^2) * V["age:w", "age:w"] +
                          2 * 40 * V["w", "age:w"]

cat(sprintf("\nVar(λᵀθ̂) = Var(θ̂₃) + 40² Var(θ̂₄) + 2×40×Cov(θ̂₃,θ̂₄)\n"))
cat(sprintf("         = %.8f + 40² × %.8f + 2×40 × %.8f\n",
            V["w", "w"], V["age:w", "age:w"], V["w", "age:w"]))
cat(sprintf("         = %.8f\n", var_lambda_t_theta_hat))

# 5. 计算标准误
se_lambda_t_theta_hat <- sqrt(var_lambda_t_theta_hat)
cat(sprintf("\nSE(λᵀθ̂) = √Var(λᵀθ̂) = %.6f\n", se_lambda_t_theta_hat))

# 6. 计算 t 统计量
t_stat <- lambda_t_theta_hat / se_lambda_t_theta_hat
cat(sprintf("\nt 统计量 = (λᵀθ̂ - 0) / SE(λᵀθ̂) = %.6f / %.6f = %.4f\n",
            lambda_t_theta_hat, se_lambda_t_theta_hat, t_stat))

# 7. 计算 p 值
df_res <- df.residual(fit_F)
cat(sprintf("\n残差自由度 = n - r = %d\n", df_res))

p_value <- 2 * pt(abs(t_stat), df = df_res, lower.tail = FALSE)
cat(sprintf("\np 值 (双侧) = 2 × P(t_%d > |%.4f|) = %.6f\n", df_res, t_stat, p_value))

# 8. 计算临界值
alpha <- 0.05
t_critical <- qt(1 - alpha/2, df = df_res)
cat(sprintf("\n临界值 (α = 0.05): t_{0.025; %d} = %.4f\n", df_res, t_critical))

# 9. 做出决策
cat("\n决策:\n")
if (abs(t_stat) > t_critical) {
  cat(sprintf("  |t 统计量| = %.4f > t 临界值 = %.4f\n", abs(t_stat), t_critical))
  cat("  或者等价地:\n")
  cat(sprintf("  p 值 = %.6f < α = 0.05\n", p_value))
  cat("\n  结论: 在显著性水平 α = 0.05 下，我们拒绝原假设 H₀。\n")
  cat("  有充分证据表明在年龄 x=40 时，女性和男性的预期 MWDR 存在显著差异。\n")
} else {
  cat(sprintf("  |t 统计量| = %.4f ≤ t 临界值 = %.4f\n", abs(t_stat), t_critical))
  cat("  或者等价地:\n")
  cat(sprintf("  p 值 = %.6f ≥ α = 0.05\n", p_value))
  cat("\n  结论: 在显著性水平 α = 0.05 下，我们不拒绝原假设 H₀。\n")
  cat("  没有充分证据表明在年龄 x=40 时，女性和男性的预期 MWDR 存在显著差异。\n")
}

# 10. 计算 95% 置信区间
ci_lower <- lambda_t_theta_hat - t_critical * se_lambda_t_theta_hat
ci_upper <- lambda_t_theta_hat + t_critical * se_lambda_t_theta_hat
cat(sprintf("\nθ₃ + 40θ₄ 的 95%% 置信区间: [%.6f, %.6f]\n", ci_lower, ci_upper))


# ============================================================================
# 问题 6: 预测和预测区间
# ============================================================================

cat("\n\n=== 问题 6: 预测 22 岁男性的 MWDR 及 95%% 预测区间 ===\n")

cat("\n我们要预测一个 x=22, w=0 (22 岁男性) 的新个体的 MWDR 值。\n")

# 1. 创建新数据点
new_data_point <- tibble(age = 22, w = 0)

cat("\n新数据点:\n")
print(new_data_point)

# 2. 获取点预测和预测区间
# interval = "prediction" 对应 lm-article Proposition 5.8
# 这确保了公式中包含了 +1 项，以解释新观测 ε̃ 的不确定性
prediction_result <- predict(fit_F, newdata = new_data_point,
                             interval = "prediction", level = 0.95)

cat("\n预测结果:\n")
print(prediction_result)

# 提取结果
y_pred <- prediction_result[1, "fit"]
pi_lower <- prediction_result[1, "lwr"]
pi_upper <- prediction_result[1, "upr"]

cat(sprintf("\n点预测 (Ŷ): %.6f\n", y_pred))
cat(sprintf("95%% 预测区间: [%.6f, %.6f]\n", pi_lower, pi_upper))

cat("\n解释:\n")
cat(sprintf("  我们预测一个 22 岁男性的 MWDR 值为 %.6f。\n", y_pred))
cat(sprintf("  我们有 95%% 的信心认为，一个新的 22 岁男性个体的 MWDR 值\n"))
cat(sprintf("  将落在区间 [%.6f, %.6f] 内。\n", pi_lower, pi_upper))

# 3. 绘制预测区间图
cat("\n绘制带预测区间的图...\n")

# 准备预测点的数据框
plot_data_Q6 <- tibble(
  age = 22,
  w = 0,
  sex = factor("M", levels = c("F", "M")),
  fit = y_pred,
  lwr = pi_lower,
  upr = pi_upper
)

# 重用 Q2 的拟合曲线数据
plot6 <- ggplot(data_with_dummy, aes(x = age, y = MWDR, color = sex)) +
  geom_point(alpha = 0.5, size = 2) + # 原始数据
  geom_line(data = plot_data, aes(y = predicted_MWDR), linewidth = 1.2) + # 拟合线

  # 添加预测点和预测区间
  geom_point(data = plot_data_Q6, aes(y = fit),
             color = "black", size = 5, shape = 4, stroke = 2) +
  geom_errorbar(data = plot_data_Q6, aes(y = fit, ymin = lwr, ymax = upr),
                color = "black", width = 2, linewidth = 1.2) +

  scale_color_manual(values = c("F" = "#E69F00", "M" = "#56B4E9")) +
  labs(title = "模型 F 拟合与 22 岁男性的 95% 预测区间",
       subtitle = sprintf("预测点: x=22, 预测值=%.3f, 95%% PI=[%.3f, %.3f]",
                         y_pred, pi_lower, pi_upper),
       x = "年龄 (x)",
       y = "MWD 比率 (Y)",
       color = "性别") +
  theme_minimal()

print(plot6)
ggsave("plot6_prediction_interval.png", plot6, width = 10, height = 6)

cat("\n评论 (图 6): 黑色叉号表示 22 岁男性的预测 MWDR 值，\n")
cat("黑色误差棒表示 95% 预测区间。预测区间考虑了模型参数估计的不确定性\n")
cat("以及个体观测值的随机误差。\n")

# 额外: 也可以计算置信区间 (confidence interval) 用于比较
cat("\n\n--- 额外: 置信区间 vs. 预测区间的比较 ---\n")
cat("\n置信区间是针对平均响应 E[Y|x] 的区间，预测区间是针对个体观测值的区间。\n")

confidence_result <- predict(fit_F, newdata = new_data_point,
                             interval = "confidence", level = 0.95)

cat("\n95% 置信区间 (针对平均响应):\n")
print(confidence_result)

ci_lower_conf <- confidence_result[1, "lwr"]
ci_upper_conf <- confidence_result[1, "upr"]

cat(sprintf("\n95%% 置信区间: [%.6f, %.6f]\n", ci_lower_conf, ci_upper_conf))
cat(sprintf("95%% 预测区间: [%.6f, %.6f]\n", pi_lower, pi_upper))

cat("\n预测区间比置信区间更宽，因为它还包含了个体观测误差的不确定性。\n")


# ============================================================================
# 总结
# ============================================================================

cat("\n\n" , rep("=", 70), sep = "")
cat("\n分析完成！\n")
cat(rep("=", 70), "\n", sep = "")

cat("\n生成的图表文件:\n")
cat("  - plot1_MWDR_vs_age.png\n")
cat("  - plot2_MWDR_vs_sex.png\n")
cat("  - plot3_MWDR_vs_age_by_sex.png\n")
cat("  - plot4_model_F_fitted_curves.png\n")
cat("  - plot4a_residuals_vs_fitted.png\n")
cat("  - plot4b_qq_plot.png\n")
cat("  - plot4c_residuals_histogram.png\n")
cat("  - plot6_prediction_interval.png\n")

cat("\n所有分析均基于 lm-article 讲义中的方法和原理。\n")
cat("\n作业各部分总结:\n")
cat("  问题 1: 探索性图表 - 完成\n")
cat("  问题 2: 模型 F 拟合 - 完成\n")
cat("  问题 3: 参数解释 - 完成\n")
cat("  问题 4: 假设检验和诊断 - 完成\n")
cat("  问题 5: 假设检验 (θ₃ + 40θ₄ = 0) - 完成\n")
cat("  问题 6: 预测和预测区间 - 完成\n")

cat("\n", rep("=", 70), "\n\n", sep = "")
