# =============================================================================
# 测试脚本 - 验证 R 环境配置
# =============================================================================

cat("\n========================================\n")
cat("测试 R 环境配置\n")
cat("========================================\n\n")

# 1. 检查 R 版本
cat("1. R 版本:\n")
print(R.version.string)

# 2. 检查工作目录
cat("\n2. 当前工作目录:\n")
print(getwd())

# 3. 检查数据文件是否存在
cat("\n3. 检查 Semaglutide.csv 是否存在:\n")
if (file.exists("Semaglutide.csv")) {
  cat("   ✓ 数据文件存在\n")
} else {
  cat("   ✗ 数据文件不存在！请确保在正确的目录中\n")
  cat("   提示: 使用 setwd() 设置工作目录\n")
}

# 4. 检查必要的包
cat("\n4. 检查必要的 R 包:\n")

check_package <- function(pkg_name) {
  if (requireNamespace(pkg_name, quietly = TRUE)) {
    cat(sprintf("   ✓ %s 已安装\n", pkg_name))
    return(TRUE)
  } else {
    cat(sprintf("   ✗ %s 未安装\n", pkg_name))
    cat(sprintf("   安装命令: install.packages(\"%s\")\n", pkg_name))
    return(FALSE)
  }
}

tidyverse_ok <- check_package("tidyverse")
broom_ok <- check_package("broom")

# 5. 尝试加载包
if (tidyverse_ok && broom_ok) {
  cat("\n5. 尝试加载包:\n")

  tryCatch({
    library(tidyverse)
    cat("   ✓ tidyverse 加载成功\n")
  }, error = function(e) {
    cat("   ✗ tidyverse 加载失败:", conditionMessage(e), "\n")
  })

  tryCatch({
    library(broom)
    cat("   ✓ broom 加载成功\n")
  }, error = function(e) {
    cat("   ✗ broom 加载失败:", conditionMessage(e), "\n")
  })

  # 6. 尝试读取数据
  if (file.exists("Semaglutide.csv")) {
    cat("\n6. 尝试读取数据:\n")

    tryCatch({
      data <- read_csv("Semaglutide.csv", show_col_types = FALSE)
      cat(sprintf("   ✓ 成功读取 %d 行数据\n", nrow(data)))
      cat(sprintf("   ✓ 列名: %s\n", paste(names(data), collapse = ", ")))

      # 显示前几行
      cat("\n   前 5 行数据:\n")
      print(head(data, 5))

    }, error = function(e) {
      cat("   ✗ 读取数据失败:", conditionMessage(e), "\n")
    })
  }

  # 7. 简单的绘图测试
  cat("\n7. 测试绘图功能:\n")

  tryCatch({
    if (exists("data")) {
      p <- ggplot(data, aes(x = age, y = MWDR)) +
        geom_point() +
        labs(title = "测试图: MWDR vs Age")

      print(p)

      # 尝试保存
      ggsave("test_plot.png", p, width = 6, height = 4)
      cat("   ✓ 绘图成功\n")
      cat("   ✓ 图表已保存为 test_plot.png\n")
    }
  }, error = function(e) {
    cat("   ✗ 绘图失败:", conditionMessage(e), "\n")
  })

} else {
  cat("\n请先安装缺失的包:\n")
  cat("install.packages(c(\"tidyverse\", \"broom\"))\n")
}

cat("\n========================================\n")
cat("测试完成！\n")
cat("========================================\n\n")

cat("如果所有测试都通过（显示 ✓），你就可以运行主分析脚本了:\n")
cat("source(\"semaglutide_analysis.R\")\n\n")
