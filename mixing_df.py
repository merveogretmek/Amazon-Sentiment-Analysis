import pandas as pd

df1 = pd.read_csv("Data/apparel.csv")
df1 = df1.head(2500)
df1 = df1[['star_rating','review_body']]

df2 = pd.read_csv("Data/automotive.csv")
df2 = df2.head(2500)
df2 = df2[['star_rating','review_body']]

df3 = pd.read_csv("Data/baby.csv")
df3 = df3.head(2500)
df3 = df3[['star_rating','review_body']]

df4 = pd.read_csv("Data/beauty.csv")
df4 = df4.head(2500)
df4 = df4[['star_rating','review_body']]

df5 = pd.read_csv("Data/camera.csv")
df5 = df5.head(2500)
df5 = df5[['star_rating','review_body']]

df6 = pd.read_csv("Data/electronics.csv")
df6 = df6.head(2500)
df6 = df6[['star_rating','review_body']]

df7 = pd.read_csv("Data/furniture.csv")
df7 = df7.head(2500)
df7 = df7[['star_rating','review_body']]

df8 = pd.read_csv("Data/grocery.csv")
df8 = df8.head(2500)
df8 = df8[['star_rating','review_body']]

df9 = pd.read_csv("Data/health_personal_care.csv")
df9 = df9.head(2500)
df9 = df9[['star_rating','review_body']]

df10 = pd.read_csv("Data/home_entertainment.csv")
df10 = df10.head(2500)
df10 = df10[['star_rating','review_body']]

df11 = pd.read_csv("Data/home_improvement.csv")
df11 = df11.head(2500)
df11 = df11[['star_rating','review_body']]

df12 = pd.read_csv("Data/lawn_and_garden.csv")
df12 = df12.head(2500)
df12 = df12[['star_rating','review_body']]

df13 = pd.read_csv("Data/luggage.csv")
df13 = df13.head(2500)
df13 = df13[['star_rating','review_body']]

df14 = pd.read_csv("Data/major_appliances.csv")
df14 = df14.head(2500)
df14 = df14[['star_rating','review_body']]

df15 = pd.read_csv("Data/mobile_electronics.csv")
df15 = df15.head(2500)
df15 = df15[['star_rating','review_body']]

df16 = pd.read_csv("Data/musical_instruments.csv")
df16 = df16.head(2500)
df16 = df16[['star_rating','review_body']]

df17 = pd.read_csv("Data/office_products.csv")
df17 = df17.head(2500)
df17 = df17[['star_rating','review_body']]

df18 = pd.read_csv("Data/outdoors.csv")
df18 = df18.head(2500)
df18 = df18[['star_rating','review_body']]

df19 = pd.read_csv("Data/personal_care_appliances.csv")
df19 = df19.head(2500)
df19 = df19[['star_rating','review_body']]

df20 = pd.read_csv("Data/watches.csv")
df20 = df20.head(2500)
df20 = df20[['star_rating','review_body']]


frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20]

dataframe = pd.concat(frames)
dataframe = dataframe.sample(frac=1).reset_index(drop=True)

print(dataframe)

dataframe.to_csv("amazon_dataframe.csv", index = False)