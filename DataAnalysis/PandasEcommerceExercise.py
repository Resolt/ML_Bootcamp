import numpy as np
import pandas as pd

dir = "Bootcamp/DataAnalysis/"

ecom = pd.read_csv(dir + "Ecommerce.csv")

print("Head of ecom: \n{}",format(ecom.head()))

print("\nRows and Columns of ecom: {}\n".format(ecom.shape))
print("Info from ecom:\n{}\n".format(ecom.info()))

print("The average purchase price: {}\n".format(ecom['Purchase Price'].mean()))

print(
	"The highest purchase price: {}\nThe lowest purchase price: {}\n".format(
		ecom['Purchase Price'].max(), ecom['Purchase Price'].min()
	)
)

print(
	"Amount of people with English as language of choice: {}\n".format(
		ecom['Language'].apply(lambda x: x == 'en').sum()
	)
)

print(
	"Amount of people with the job title lawyer: {}\n".format(
		ecom['Job'].apply(lambda x: x.lower() =="lawyer").sum()
	)
)

print(
	"AM purchases: {}\nPM purhcases: {}\n".format(
		ecom['AM or PM'].apply(lambda x: x == "AM").sum(),
		ecom['AM or PM'].apply(lambda x: x == "PM").sum()
	)
)

# alternative
print("AM/PM purchases:\n{}\n".format(ecom['AM or PM'].value_counts()))

print(
	"The 5 most common job titles: \n{}\n".format(
		ecom['Job'].value_counts().head(5)
	)
)

print(
	"Purchase price of Lot: 90 WT purchase: {}\n".format(
		ecom['Purchase Price'][ecom['Lot'] == "90 WT"]
	)
)

print(
	"Email of person with CC no: {}\n".format(
		ecom['Email'][ecom['Credit Card'] == 4926535242672853]
	)
)

print(
	"Amount of people with AmEx as provider and purhcase above 95 bucks made: {}\n".format(
		np.sum((ecom['CC Provider'] == "American Express") & (ecom['Purchase Price'] > 95))
	)
)

print(
	"Amount of people with a credit card that expires in 2025: {}\n".format(
		np.sum(ecom['CC Exp Date'].apply(lambda x : x.split('/')[1] == '25'))
	)
)

print(
	"Top 5 most popular email domains:\n{}\n".format(
		ecom['Email'].apply(lambda x: x.split('@')[1]).value_counts().head(5)
	)
)
