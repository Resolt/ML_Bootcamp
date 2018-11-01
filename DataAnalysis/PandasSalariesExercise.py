import numpy as np
import pandas as pd

dir = "Bootcamp/DataAnalysis/"

sal = pd.read_csv(dir + "Salaries.csv")

print(sal.head())
print(sal.info())

print("\nAverage base pay: {}\n".format(sal['BasePay'].mean()))

print("Highest over time pay: {}\n".format(sal['OvertimePay'].max()))

print(
	"JOSEPH DRISCOLL's job title: {}\n".format(
		sal['JobTitle'][sal['EmployeeName'] == "JOSEPH DRISCOLL"]
	)
)

print(
	"JOSEPH DRISCOLL's total pay including benefits: {}\n".format(
		sal['TotalPayBenefits'][sal['EmployeeName'] == "JOSEPH DRISCOLL"]
	)
)

print(
	"Lowest Paid Person: {}\n".format(
		sal['EmployeeName'][sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()]
	)
)

print(
	"Average Base Pay Per Year:\n{}\n".format(
		sal.groupby('Year')['BasePay'].mean()
	)
)

print(
	"Amount of Unique Job Titles: {}\n".format(
		sal['JobTitle'].nunique()
	)
)

print(
	"The top 5 most common jobs:\n{}\n".format(
		sal['JobTitle'].value_counts().head(5)
	)
)

print(
	"Amount of job titles represented by just 1 person in 2013: {}\n".format(
		sal['JobTitle'][sal['Year'] == 2013].value_counts().apply(lambda x : x == 1).sum()
	)
)

print(
	"Amount of people having the word 'Chief' in their job title: {}\n".format(
		sal['JobTitle'].apply(lambda x: "chief" in x.lower().split() != -1).sum()
	)
)

print(
	"Correlation between length of the job title string and the salary: {}\n".format(
		np.corrcoef(np.array((sal['JobTitle'].apply(lambda x: len(x)), sal['TotalPayBenefits'])))
	)
)




