# Packages Required
# ggplot2, stringr
install.packages(c("ggplot2", "stingr"))
library(ggplot2)
library(stringr)
setwd("C:\\Users\\bangxi\\Desktop\\lending-club")
data_accept <- read.csv("accepted_2007_to_2018Q3.csv", header = T, stringsAsFactors = F)
data_obs <- read.csv("accepted_2007_to_2018Q3.csv", header = T, stringsAsFactors = F)
str(data_accept)
nrow(data_accept)
lf <- function(col)
{
  return(
    levels(factor(data_accept[, col]))
  )
}

fc <- function(name)
{
  return(
    which(colnames(data_accept) %in% name)
  )
}

cat2mat <- function(name)
{
  level_tmp <- lf(name)
  data <- data_accept[, name]
  mat <- c()
  for(i in 1:length(level_tmp))
  {
    mat <- cbind(mat, 
             as.matrix(ifelse(data %in% level_tmp[i], 
                                       1, 0))
             )
  }
  colnames(mat) <- level_tmp
  return(mat)
}

fna <- function(name)
{
  return(
    which(is.na(data_accept[, name]))
  )
}
# Fund amount inventory
length(data_accept$funded_amnt_inv >= 0) == nrow(data_accept)

# Term
levels(factor(data_accept$term))

## Delete the term with ""
data_accept <- 
  data_accept[- which(data_accept$term == ""), ]
data_accept$term_int <- 
  as.numeric(unlist(strsplit(data_accept$term, " months")))

# Installment 
all(data_accept$installment > 0)

# Grade - delete
data_accept <- data_accept[, -9]

# Sub Grade 
sub_gd_df <- as.data.frame(cat2mat('sub_grade'))
colnames(sub_gd_mat) <- sub_gd_level
data_accept <- cbind(data_accept, sub_gd_df)
fc("sub_grade")
data_accept <- data_accept[, -9]
rm("sub_gd_tmp", "sub_gd_mat")

# EMP_title - delete
data_accept$emp_title
fc("emp_title")
data_accept <- data_accept[, -9]

# Home-ownership
home_own_df <- as.data.frame(cat2mat("home_ownership"))
data_accept <- cbind(data_accept, home_own_df)
fc('home_ownership')
data_accept <- data_accept[, -10]

# Emp-length
data_accept$emp_length[which(data_accept$emp_length == "")] = 
  "0 years"
lf("emp_length")
emp_length_df <- as.data.frame(cat2mat("emp_length"))
fc("emp_length")
data_accept <- data_accept[, -9]
dim(emp_length_df)
data_accept <- cbind(data_accept, emp_length_df)

# verification_status
data_accept$verification_status
lf("verification_status")
vs_df <- as.data.frame(cat2mat("verification_status"))
fc("verification_status")
data_accept <- data_accept[, -10]
dim(vs_df)
data_accept <- cbind(data_accept, vs_df)

# issue_d
undecide <- data_accept$issue_d
fc("issue_d")
data_accept <- data_accept[, -10]

# Loan status - no 'na' value
loan_stat_level = data_accept$loan_status
data_accept <- data_accept[-which(data_accept$loan_status == "Current"), ]
length(which(data_accept$loan_status == "Current"))
lf("loan_status")

# Payment plan - delete
data_accept$ptmnt_plan
fc("pymnt_plan")
data_accept <- data_accept[, -11]

# url - delete
data_accept$url
fc("url")
data_accept <- data_accept[, -11]

# Desc - delete
fc("desc")
data_accept <- data_accept[, -11]

# Purpose
# data_accept$purpose
lf("purpose")
purpose_df <- as.data.frame(cat2mat("purpose"))
data_accept <- data_accept[, -fc("purpose")]
data_accept <- cbind(data_accept, purpose_df)

# title
str(data_accept)
data_accept <- data_accept[, -fc("title")]

# zip
# data_accept$zip_code
data_accept <- data_accept[, -fc("zip_code")]

# Address-state

# Dti

lf("dti")

# Delin_2yrsc
data_accept$delinq_2yrs
lf("delinq_2yrs")

# Earliest
# data_accept$earliest_cr_line
data_accept <- data_accept[, -fc("earliest_cr_line")]

# FICO-range-low
lf("fico_range_low")

# FICO-range-high
# data_accept$fico_range_high
data_accept <- data_accept[, -fc("fico_range_high")]

# Inq-last-6months
lf("inq_last_6mths")

# Month since last delinq
str(data_accept)
which(is.na(data_accept$mths_since_last_delinq))
data_accept$mths_since_last_delinq[which(
  is.na(data_accept$mths_since_last_delinq))] = 2000
which(is.na(data_accept$mths_since_last_delinq))
lf("mths_since_last_delinq")

# Month since last record
which(is.na(data_accept$mths_since_last_record))
data_accept$mths_since_last_record[which(
  is.na(data_accept$mths_since_last_record))] = 2000
which(is.na(data_accept$mths_since_last_record))

# Revol_balance
which(is.na(data_accept$revol_bal))

# Total
which(is.na(data_accept$total_acc))
all(data_accept$total_acc >= 0) # Containing NA values
which(data_accept$total_acc == 0)
data_accept$total_acc[which(
  is.na(data_accept$total_acc))] = 0

# init_list_status
data_accept$initial_list_status
lf("initial_list_status")
fna('initial_list_status')
ils_df <- cat2mat("initial_list_status")
data_accept <- cbind(data_accept, ils_df)
data_accept <- data_accept[, -fc("initial_list_status")]

# Last payment amount 
data_accept$last_pymnt_amnt
fna("last_pymnt_amnt")

# Last payment day
data_accept$last_pymnt_d
data_accept <- data_accept[, -fc("last_pymnt_d")]

# Next payment day
data_accept$next_pymnt_d
data_accept <- data_accept[, -fc("next_pymnt_d")]

# Last credit pull
data_accept$last_credit_pull_d
data_accept <- data_accept[, -fc("last_credit_pull_d")]

# Last fico
data_accept$last_fico_range_high
fna("last_fico_range_high")

# Collections 12 months ex med
data_accept$collections_12_mths_ex_med
lf("collections_12_mths_ex_med")
na_tmp <- fna("collections_12_mths_ex_med")
data_accept <- data_accept[-na_tmp, ]

# Months since last delete, Open_... delete, total bal_il delete, 
# ll_until, open_tv delete, acc_open_recent_bc delete, 
data_accept$num_il_tl
data_accept <- data_accept[, -fc(c('mths_since_last_major_derog', 'mths_since_rcnt_il',
                                   'mths_since_recent_bc', 'num_accts_ever_120_pd', 
                                   'mths_since_recent_revol_delinq', 'total_bal_il',
                                   'policy_code', 'annual_inc_joint', 'dti_joint', 
                                   'verification_status_joint', 'acc_now_delinq', 
                                   'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m',
                                   'all_util', 'open_rv_12m', 'open_rv_24m', 'acc_open_past_24mths',
                                   'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 
                                   'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 
                                   'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats',
                                     'num_il_tl'))]
data_accept <- data_accept[, -fc("il_util")]

# Member ud
data_accept$member_id
data_accept <- data_accept[, -fc("member_id")]

# clear NA data
colnames(data_accept)
data_accept <- data_accept[, -fc(c("sec_app_fico_range_low", "sec_app_fico_range_high",                   
"sec_app_earliest_cr_line", "sec_app_inq_last_6mths",                    
"sec_app_mort_acc", "sec_app_open_acc",                          
"sec_app_revol_util", "sec_app_open_act_il",                       
"sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths",          
"sec_app_collections_12_mths_ex_med", "sec_app_mths_since_last_major_derog"))]

# Harship - remain hardship_flag
data_accept <- data_accept[, -fc(c("hardship_type" ,                            
                    "hardship_reason","hardship_status",                           
                    "deferral_term",                              "hardship_amount"   ,                        
                    "hardship_start_date" ,                       "hardship_end_date"  ,                       
                    "payment_plan_start_date"    ,                "hardship_length"     ,                      
                    "hardship_dpd"         ,                      "hardship_loan_status"    ,                  
                    "hardship_payoff_balance_amount" ,           
                    "hardship_last_payment_amount"))]

# settlement delete
data_accept <- data_accept[, -fc(c("disbursement_method" ,                 
                                   "debt_settlement_flag_date"  ,               
                                   "settlement_date"             ,     "settlement_amount"   ,                      
                                   "settlement_percentage"      ,     "settlement_term"))]


# other delete
data_accept <- data_accept[, -fc('revol_bal_joint')]

# Original projected additional accrued
data_accept[which(is.na(data_accept$orig_projected_additional_accrued_interest)), ]$orig_projected_additional_accrued_interest = 0

# month since recent bc dlq
data_accept[which(is.na(data_accept$mths_since_recent_bc_dlq)), ]$mths_since_recent_bc_dlq = 2000
data_accept[which(is.na(data_accept$mths_since_recent_inq)), ]$mths_since_recent_inq = 2000

# application type 2 binary
data_accept <- cbind(data_accept, cat2mat("application_type"))
data_accept <- data_accept[, -fc("application_type")]

# hardship_flag
data_accept <- cbind(data_accept, cat2mat("hardship_flag"))
data_accept <- data_accept[, -fc("hardship_flag")]

# debt_settlement_flag
data_accept <- data_accept[, -fc("debt_settlement_flag")]

# 
data_accept[data_accept$settlement_status == "", ]$settlement_status = "No"
data_accept <- cbind(data_accept, cat2mat("settlement_status"))
data_accept <- cbind(data_accept, cat2mat("addr_state"))
data_accept <- data_accept[, -fc("term")]
data_accept <- data_accept[, -fc("addr_state")]
data_accept <- data_accept[, -fc("settlement_status")]
data_accept <- data_accept[, -fc("id")]

# predict value combining
data_accept[data_accept$loan_status == "Does not meet the credit policy. Status:Charged Off", ]$loan_status = "Charged Off"
data_accept[data_accept$loan_status == "Does not meet the credit policy. Status:Fully Paid", ]$loan_status = "Fully Paid"

# sub-grade



# Examine through every column finding the NA value
navalue <- c()
for(i in 1:ncol(data_accept))
{
  if(length(which(is.na(data_accept[, i]))) >= 1)
  {
    navalue <- c(navalue, i)
  }
}
nacolnames <- colnames(data_accept)[navalue]
narow <- c()
each_na <- c()
for(i in nacolnames)
{
  narow <- c(narow, which(is.na(data_accept[, i])))
  each_na <- c(each_na, length(which(is.na(data_accept[, i]))))
}
each_na
narow <- unique(narow)
length(narow)

nacolnames
each_na
narow
# Remove row that contains NA dti
data_accept <- data_accept[-which(is.na(data_accept$dti)), ]

# Remove NA values of inq_last_6mths
data_accept <- data_accept[-which(is.na(data_accept$inq_last_6mths)), ]

# Remove NA values of revol_util
data_accept <- data_accept[-which(is.na(data_accept$revol_util)), ]

# 
which(is.na(data_accept$tot_coll_amt))


