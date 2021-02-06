import pandas as pd

wireless_df = pd.read_csv("amazon_wireless_df.csv")
wireless_df = wireless_df.tail(100000)
wireless_df = wireless_df[['star_rating','review_body']]
print(wireless_df)


software_df = pd.read_csv("amazon_software_df.csv")
software_df = software_df.tail(100000)
software_df = software_df[['star_rating','review_body']]
print(software_df)


music_df = pd.read_csv("amazon_music_df.csv")
print(music_df)
music_df = music_df.tail(100000)
music_df = music_df[['star_rating','review_body']]


frames = [wireless_df, software_df, music_df]

df = pd.concat(frames)
df.to_csv("amazon_data2.csv", index= False, encoding="utf-8")