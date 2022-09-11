import os
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Load the data sets
    exp_dir = 'PSA2KNW'
    alogs = pd.read_csv(os.path.join(exp_dir, 'exp_alogs.csv'))
    plogs = pd.read_csv(os.path.join(exp_dir, 'exp_plogs.csv'))
    slogs = pd.read_csv(os.path.join(exp_dir, 'exp_slogs.csv'))
    priors = pd.read_csv(os.path.join(exp_dir, 'priors.csv'))
    # -----------------------------------------


    # -----------------------------------------
    # TASK 1

    # Calculate the number of students in Control/Treatment
    control_count = len(alogs[alogs['assigned_condition'] == 'Control'])
    treatment_count = len(alogs[alogs['assigned_condition'] == 'Treatment'])
    print(f"# of students in control condition: {control_count}")
    print(f"# of students in treatment condition: {treatment_count}")

    # Calculate the proportion of students who completed the assignment while in Control/Treatment
    control_completed_count = len(alogs[(alogs['assigned_condition'] == 'Control') & (pd.notna(alogs['end_time']))])
    treatment_completed_count = len(alogs[(alogs['assigned_condition'] == 'Treatment') & (pd.notna(alogs['end_time']))])
    print(f"% of students placed in a control condition who completed the assignment: {control_completed_count / control_count}")
    print(f"% of students placed in a treatment condition who completed the assignment: {treatment_completed_count / treatment_count}")

    # To find the likelihood that a student completed their assignment was influenced by which condition they were placed it,
    # we can utilize the two-proportion z-test to determine whether the two populations (Control/Treatment) are statistically different.
    # Our null hypothesis will be that the two populations are equal with the common standard 0.05 significance level.
    # I believe that z-test is appropriate because the test attempts to determine whether two population is different or not.
    # In this case, the two populations are students in Control/Treatment conditions with the proportion being students that have actually completed their assignment.

    # Declare the number of observations (total population)
    num_obs = np.array([control_count, treatment_count])
    # Declare the count of student who completed the assignment in Control/Treatment
    completed_count = np.array([control_completed_count, treatment_completed_count])
    # Calculate the z-statistic and the corresponding p-value using statsmodels library
    z_stat, p_val = proportions_ztest(completed_count, num_obs)
    print(f"z-statistic score: {z_stat}")
    print(f"two-tailed hypothesis p-value: {p_val}\n")

    # The two-tailed hypothesis p-value of 0.63 is not statistically significant (0.63 > 0.05) and therefore, we cannot reject the null hypothesis.
    # This means that the two-proportion z-test suggests that there is no significant relationship between the likelihood of students completing their assignment,
    # and the condition that they were placed in.
    # One explanation for this result is that there were external factors at play, such as parents urging their children to finish the assignment.
    # This would mean that regardless of the assigned conditions of the student, they had to finish the assignment anyway.
    # In this case, the assignment completion rate may not be the best predictor of whether or not the treatment is effective.
    # -----------------------------------------


    # -----------------------------------------
    # TASK 2

    # 3 features that I think could influence the students' ability to complete their assignments are the following:
    # 1. problem_set_percent_completed (student_prior_completed_problem_set_count / student_prior_started_problem_set_count)
    #    because this feature can indicate how responsible a student is in completing his/her assignments.
    # 2. class_prior_average_correctness because this feature can indicate how well the current class' in-class teaching method is working for the students.
    # 3. opportunity_zone because this feature can indicate the amount of resources that is given to the student.

    # Pull only relevant columns from alogs and priors tables
    relevant_alogs = alogs[['student_id', 'end_time', 'assigned_condition']]
    relevant_priors = priors[['student_id',
                              'student_prior_started_problem_set_count',
                              'student_prior_completed_problem_set_count',
                              'class_prior_average_correctness',
                              'opportunity_zone']]
    # Create a features tables with only relevant features
    features = pd.merge(relevant_alogs, relevant_priors, on='student_id')

    # Cleaning up the features table to include only relevant features
    # Create a one-hot column for whether or not a student completed their assignment based on NaN values in end_time
    features['assignment_completed'] = np.where(pd.notna(features['end_time']), 1, 0)
    # Clean up opportunity_zone column values by replacing 'Yes' with 1 and 'No' with 0
    features.replace({'opportunity_zone' : {'Yes' : 1, 'No' : 0}}, inplace=True)
    # Clean up assigned_condition column values by replacing 'Treatment' with 1 and 'Control' with 0
    features.replace({'assigned_condition' : {'Treatment' : 1, 'Control' : 0}}, inplace=True)
    # Create a problem_set_percent_completed column based on the ratio between problem set completion / problem set started.
    features['problem_set_percent_completed'] = features['student_prior_completed_problem_set_count'] / features['student_prior_started_problem_set_count']
    # Drop unused columns
    features.drop(columns=['end_time',
                           'student_prior_started_problem_set_count',
                           'student_prior_completed_problem_set_count'], inplace=True)

    # Feature 1: problem_set_percent_completed
    # We only consider students who have started a problem set.
    all_problem_set = features[pd.notna(features['problem_set_percent_completed'])]
    control_problem_set = features[(pd.notna(features['problem_set_percent_completed'])) & (features['assigned_condition'] == 0)]
    treatment_problem_set = features[(pd.notna(features['problem_set_percent_completed'])) & (features['assigned_condition'] == 1)]

    # Calculate the correlation between assignment completion and the problem set percent complete
    correlation1_all = all_problem_set['problem_set_percent_completed'].corr(all_problem_set['assignment_completed'])
    correlation1_control = control_problem_set['problem_set_percent_completed'].corr(control_problem_set['assignment_completed'])
    correlation1_treatment = treatment_problem_set['problem_set_percent_completed'].corr(treatment_problem_set['assignment_completed'])

    print("Correlation between assignment completion and feature 1 (problem_set_percent_completed)")
    print(f"Correlation for all students: {correlation1_all}")
    # The correlation between assignment completion and the problem set percent complete for all students is 0.24.
    # This suggests that there is a minor positive correlation but not significant one to be a sole predictor.

    print(f"Correlation for control students: {correlation1_control}")
    # The correlation between assignment completion and the problem set percent complete for control students is 0.03.
    # This suggests that there is no significant correlation between the two variables.

    print(f"Correlation for treatment students: {correlation1_treatment}\n")
    # The correlation between assignment completion and the problem set percent complete for treatment students is 0.17.
    # This suggests that there is a minor positive correlation but not significant one to be a sole predictor.
    # However, problem set percent complete may be a better predictor for students in treatment than students in control setting.

    # Feature 2: class_prior_average_correctness
    # We only consider non-NaN values for the class average correctness
    all_class_correctness = features[pd.notna(features['class_prior_average_correctness'])]
    control_class_correctness = features[(pd.notna(features['class_prior_average_correctness'])) & (features['assigned_condition'] == 0)]
    treatment_class_correctness = features[(pd.notna(features['class_prior_average_correctness'])) & (features['assigned_condition'] == 1)]

    # Calculate the correlation between assignment completion and the class average correctness
    correlation2_all = all_class_correctness['class_prior_average_correctness'].corr(all_class_correctness['assignment_completed'])
    correlation2_control = control_class_correctness['class_prior_average_correctness'].corr(control_class_correctness['assignment_completed'])
    correlation2_treatment = treatment_class_correctness['class_prior_average_correctness'].corr(treatment_class_correctness['assignment_completed'])

    print("Correlation between assignment completion and feature 2 (class_prior_average_correctness)")
    print(f"Correlation for all students: {correlation2_all}")
    # The correlation between assignment completion and the problem set percent complete for all students is 0.17.
    # This suggests that there is a minor positive correlation but not significant one to be a sole predictor.

    print(f"Correlation for control students: {correlation2_control}")
    # The correlation between assignment completion and the problem set percent complete for control students is 0.17.
    # This suggests that there is a minor positive correlation but not significant one to be a sole predictor.

    print(f"Correlation for treatment students: {correlation2_treatment}\n")
    # The correlation between assignment completion and the problem set percent complete for treatment students is 0.07.
    # This suggests that there is no significant correlation between the two variables.
    # Given these three correlations, this may suggests that class class prior average correctness
    # may be a better predictor for students in control than students in treatment setting.

    # Feature 3: opportunity_zone
    # We only consider non-NaN values for opportunity zones
    all_opportunity_zone = features[pd.notna(features['opportunity_zone'])]
    control_opportunity_zone = features[(pd.notna(features['opportunity_zone'])) & (features['assigned_condition'] == 0)]
    treatment_opportunity_zone = features[(pd.notna(features['opportunity_zone'])) & (features['assigned_condition'] == 1)]

    # Calculate the correlation between assignment completion and the opportunity zone
    correlation3_all = all_opportunity_zone['opportunity_zone'].corr(all_opportunity_zone['assignment_completed'])
    correlation3_control = control_opportunity_zone['opportunity_zone'].corr(control_opportunity_zone['assignment_completed'])
    correlation3_treatment = treatment_opportunity_zone['opportunity_zone'].corr(treatment_opportunity_zone['assignment_completed'])

    print("Correlation between assignment completion and feature 3 (opportunity_zone)")
    print(f"Correlation for all students: {correlation3_all}")
    # The correlation between assignment completion and the problem set percent complete for all students is -0.09.
    # This suggests that there is no significant correlation between the two variables.

    print(f"Correlation for control students: {correlation3_control}")
    # The correlation between assignment completion and the problem set percent complete for control students is -0.32.
    # This suggests that there is a minor negative correlation but not significant one to be a sole predictor.

    print(f"Correlation for treatment students: {correlation3_treatment}\n")
    # The correlation between assignment completion and the problem set percent complete for treatment students is -0.03.
    # This suggests that there is no significant correlation between the two variables.
    # These correlations suggest that opportunity zone may be a better predictor for students in control than sutdents in treatment setting.
    # This could be interpreted that the treatment allows students in opportunity zone to perform as good as students that aren't.

    # Show plot
    # plt.scatter(all_problem_set['problem_set_percent_completed'], all_problem_set['assignment_completed'])
    # plt.scatter(all_class_correctness['class_prior_average_correctness'], all_class_correctness['assignment_completed'])
    # plt.scatter(all_opportunity_zone['opportunity_zone'], all_opportunity_zone['assignment_completed'])
    # plt.show()
    # -----------------------------------------


    # -----------------------------------------
    # TASK 3

    # We will need to split the data into training and testing set to make sure that our model will not overfit.
    # The training set will be used to train the model. Once trained, the model will be evaluate based on how well it can predict the results on the testing set.

    # For the supervised learning model, I will use a logistic regression model because
    # this is a     regression problem where the output will either be 1 or 0 (assignment completed vs not completed).

    # Before we can train the model, we first need to clean up the data that we have.
    # 1. We convert all end_time column to a one-hot assignment_completed based on whether it contains a NaN value.
    # 2. We convert all assigned condition into one-hot column with 1 being Treatment and 0 being Control
    # 3. We drop all unused columns (end_time, student_prior_started_problem_set_count, student_prior_completed_problem_set_count)
    # 4. We drop all rows with assigned_condition that are Not Assigned.
    # 5. We drop all rows with problem_set_percent_completed that are NaN
    # 6. We drop all rows with class_prior_average_correctness that are NaN
    # 7. We drop all rows with opportunity_zone that are NaN
