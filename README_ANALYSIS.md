# Semaglutide Data Analysis - MATH48011 Midterm Coursework

## Overview

This repository contains a complete R analysis for the MATH48011 midterm coursework on Semaglutide data using the **tidyverse** framework. All methods strictly follow the **lm-article** lecture notes.

## Files

- **semaglutide_analysis.R**: Complete R script for all analysis tasks
- **Semaglutide.csv**: Original dataset (282 observations)
- **分析说明.md**: Detailed instructions in Chinese
- **IHC_lm-article-2025.pdf**: Course lecture notes (theoretical foundation)

## Quick Start

### Prerequisites

Install required R packages:

```r
install.packages("tidyverse")
install.packages("broom")
```

### Run the Analysis

**Option 1: RStudio**
1. Open `semaglutide_analysis.R` in RStudio
2. Set working directory to the project folder
3. Click "Source" or press Ctrl+Shift+S (Windows) / Cmd+Shift+S (Mac)

**Option 2: Command Line**
```bash
cd /path/to/MATH48011_midtermcoursework
Rscript semaglutide_analysis.R
```

### Output

The script will:
1. Print all analysis results to console
2. Generate 8 PNG plot files in the current directory

## Generated Plots

1. **plot1_MWDR_vs_age.png**: Scatter plot of MWDR vs. age
2. **plot2_MWDR_vs_sex.png**: Boxplot of MWDR vs. sex
3. **plot3_MWDR_vs_age_by_sex.png**: MWDR vs. age grouped by sex (with quadratic fits)
4. **plot4_model_F_fitted_curves.png**: Model F fitted curves
5. **plot4a_residuals_vs_fitted.png**: Residuals vs. fitted values (linearity & homoscedasticity check)
6. **plot4b_qq_plot.png**: Q-Q plot of residuals (normality check)
7. **plot4c_residuals_histogram.png**: Histogram of residuals
8. **plot6_prediction_interval.png**: Fitted curves with 95% prediction interval

## Analysis Structure

### Question 1: Exploratory Plots
- Three plots showing relationships between MWDR, age, and sex
- Preliminary assessment of linearity, interaction effects

### Question 2: Fit Model F
- Model: `E[Y] = θ₀ + θ₁x + θ₂x² + θ₃w + θ₄xw`
- Variables:
  - `Y`: MWDR (Mean Weekly Dose Ratio)
  - `x`: age
  - `w`: sex dummy variable (0=Male, 1=Female)
- Outputs: coefficient table, fitted equation, fitted curves

### Question 3: Interpret Parameters
- Interpretation of θ₀, θ₀+θ₃, θ₄ based on model structure

### Question 4: Assumptions and Diagnostics
- Four key assumptions for inference (based on lm-article):
  1. Correct structural form
  2. Constant variance (homoscedasticity)
  3. Independence
  4. Normality
- Graphical diagnostics: residual plots, Q-Q plot

### Question 5: Hypothesis Testing
- Test: H₀: θ₃ + 40θ₄ = 0
- Interpretation: At age x=40, is there a difference in expected MWDR between males and females?
- Method: t-test based on lm-article Proposition 5.4
- Outputs: test statistic, p-value, decision, 95% CI

### Question 6: Prediction and Prediction Interval
- Predict MWDR for a 22-year-old male (x=22, w=0)
- Method: Based on lm-article Proposition 5.8
- Outputs: point prediction, 95% prediction interval, visualization
- Bonus: Comparison between confidence interval and prediction interval

## Model F Details

### Model Specification

```
E[Y] = θ₀ + θ₁x + θ₂x² + θ₃w + θ₄xw
```

Where:
- **x**: age (continuous)
- **w**: sex dummy (0=Male, 1=Female)
- **Y**: MWDR

### Parameter Interpretations

- **θ₀**: Expected MWDR for 0-year-old males
- **θ₁**: Linear age effect (for males)
- **θ₂**: Quadratic age effect (same for both sexes)
- **θ₃**: Sex difference at age 0 (intercept difference)
- **θ₄**: Interaction effect (sex difference in linear age effect)

### R Formula

```r
lm(MWDR ~ age + I(age^2) + w + age:w, data = data_with_dummy)
```

## Theoretical Foundation

All methods are based on **lm-article (IHC_lm-article-2025.pdf)**:

- **Exploratory plots**: Section 1.2.2 (graphical methods)
- **Dummy variables**: Section 1.6.1 (categorical variables)
- **Quadratic terms**: Section 1.3.4 (polynomial regression)
- **Interaction terms**: Section 1.7 (ANCOVA)
- **Model diagnostics**: Section 3.4 (residual diagnostics)
- **Hypothesis testing**: Section 5.2.1 + Proposition 5.4 (t-test for linear combinations)
- **Prediction intervals**: Section 5.4 + Proposition 5.8 (prediction interval formula)

## Key Statistical Concepts

### Confidence Interval vs. Prediction Interval

- **Confidence Interval**: For estimating the mean response E[Y|x]
  - Narrower
  - Only accounts for parameter estimation uncertainty

- **Prediction Interval**: For predicting a new individual observation
  - Wider
  - Accounts for parameter uncertainty + individual error

Formula (lm-article Proposition 5.8):
```
PI: ŷ ± t_{α/2;n-r} × ŝ√(1 + x̃ᵀ(XᵀX)⁻¹x̃)
                         ↑
                    extra 1 term
```

### Interaction Effect

The interaction term θ₄ indicates whether the age effect on MWDR differs between males and females:

- For males (w=0): `∂E[Y]/∂x = θ₁ + 2θ₂x`
- For females (w=1): `∂E[Y]/∂x = (θ₁ + θ₄) + 2θ₂x`

If θ₄ ≠ 0, the age effect differs between sexes (interaction present).

## Tidyverse Advantages

1. **Pipe operator** `%>%`: Cleaner code flow
2. **broom::tidy()**: Convert model outputs to tidy data frames
3. **broom::augment()**: Extract fitted values and residuals easily
4. **ggplot2**: Powerful and flexible visualization

Example:
```r
data %>%
  mutate(w = ifelse(sex == "F", 1, 0)) %>%
  lm(MWDR ~ age + I(age^2) + w + age:w, data = .) %>%
  tidy()
```

## Troubleshooting

If you encounter errors:
1. Ensure all required packages are installed
2. Set the correct working directory
3. Check R version (recommend R ≥ 4.0.0)
4. Read console error messages carefully
5. Ensure `Semaglutide.csv` is in the same directory as the script

## Data Description

- **Sample size**: n = 282 observations
- **Variables**:
  - `MWDR`: Mean Weekly Dose Ratio (continuous, response variable)
  - `age`: Age in years (continuous, 18-74)
  - `sex`: Sex (categorical, "M"=Male, "F"=Female)
- **Groups**: 141 males, 141 females

## License

This is a coursework project for MATH48011. All analysis follows the methodology described in the course lecture notes (lm-article).

## Author

Created for MATH48011 Midterm Coursework using tidyverse and broom packages.

---

For detailed Chinese instructions, see **分析说明.md**.
