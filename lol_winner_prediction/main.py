import pandas as pd
import streamlit as st
from streamlit_extras.let_it_rain import rain
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


@st.cache_data
def load_dataset():
    df = pd.read_csv('lol_winner_prediction/datasets/high_diamond_ranked_10min.csv')
    return df


@st.cache_data
def load_model():
    model = joblib.load('lol_winner_prediction/lol_prediction.joblib')
    return model


st.set_page_config(layout='wide', page_icon='icon.png', page_title="League of Legends Project")
st.title(":blue[League of Legends Maç Tahmin Modeli]")
main_page, data_page, model_page = st.tabs(["Ana Sayfa", "Veri Seti", "Model"])


def main(main_page: st.delta_generator.DeltaGenerator):
    information_container = main_page.container()
    col1, col2, col3 = information_container.columns(3)
    col1.write(' ')
    col2.write("<div style='text-align: center;'><h3>League of Legends nedir?</h3></div>", unsafe_allow_html=True)
    col2.image("lol_winner_prediction\lol_cover.jpg", use_column_width=True)
    col2.markdown("""
            <div>
                <p style='font-size: 20px; font-style: italic;'>
                        League of Legends, beşe beş şeklinde oynanan, amacımızın rakibin ana üssünde bulunan merkezi imha etmek olan MOBA türünde çevrimiçi bir oyundur.
                    MOBA türünde olan bu oyun dünya çapında yüz milyondan fazla oyuncuya ev sahipliği yapmaktadır.
                    Bu uygulamada League of Legends'ın rekabetçi sisteminde bir küme olan Elmas kümesinin ilk 10 dakikasında oyuncular, kendilerine
                    avantaj sağlayan gerekliliklerin ne kadarını yaptığına göre maçı kazanıp-kazanmadığını tahmin eden model geliştirilmiştir.
                </p>
            </div>
        """, unsafe_allow_html=True)
    col3.write(' ')


def data(data_page: st.delta_generator.DeltaGenerator):
    df = load_dataset()
    data_page.dataframe(df, use_container_width=True)
    data_page.divider()
    data_page_col1, data_page_col2 = data_page.columns(2)

    fig = plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="blueKills")
    data_page_col1.subheader("Mavi Takım'ın Öldürme Dağılımı")
    data_page_col1.pyplot(fig)

    fig2 = plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="redKills")
    data_page_col2.subheader("Kırmızı Takım'ın Öldürme Dağılımı")
    data_page_col2.pyplot(fig2)

    fig3 = plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="blueKills", y="redKills", hue="blueWins")
    data_page_col1.subheader("Öldürmeye Göre Kazanma")
    data_page_col1.pyplot(fig3)

    fig4 = plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="blueAssists", y="redAssists", hue="blueWins")
    data_page_col2.subheader("Destek Skoruna Göre Kazanma")
    data_page_col2.pyplot(fig4)

    fig5 = plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="blueWardsPlaced")
    data_page_col1.subheader("Mavi Takımın Yerleştirdiği Totem Sayısı")
    data_page_col1.pyplot(fig5)

    fig6 = plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="redWardsPlaced")
    data_page_col2.subheader("Kırmızı Takımın Yerleştirdiği Totem Sayısı")
    data_page_col2.pyplot(fig6)

    fig7 = plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="blueFirstBlood", y="blueWins")
    data_page_col1.subheader("İlk Öldürme Skorunun Kazanmaya Katkısı")
    data_page_col1.pyplot(fig7)

    fig8 = plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="blueDragons", y="blueWins")
    data_page_col2.subheader("Öldürülen Ejderhanın Kazanmaya Katkısı")
    data_page_col2.pyplot(fig8)


def model(model_page: st.delta_generator.DeltaGenerator):
    user_input_col1, user_input_col2, user_input_col3, result_col = model_page.columns([1, 1, 1, 1])

    user_input_col1.subheader("Mavi Takım")
    blue_kills = user_input_col1.number_input("Öldürme", load_dataset()["blueKills"].min(),
                                              load_dataset()["blueKills"].max(), key="m1")

    blue_assists = user_input_col1.number_input("Asist", load_dataset()["blueAssists"].min(), blue_kills * 4, key="m2")

    blue_towers_destroyed = user_input_col1.number_input("Yıkılan Kule", load_dataset()["blueTowersDestroyed"].min(),
                                                         4, key="m3")
    blue_total_minions_killed = user_input_col1.number_input("Toplam Öldürülen Minyon", 0,
                                                             load_dataset()["blueTotalMinionsKilled"].max(), key="m4")
    blue_total_jungle_minions_killed = user_input_col1.number_input("Toplam Öldürülen Orman Canavarı", 0,
                                                                    load_dataset()[
                                                                        "blueTotalJungleMinionsKilled"].max(), key="m5")
    blue_total_gold = user_input_col1.number_input("Mavi Takım Toplam Altın", 0,
                                                  load_dataset()["blueTotalGold"].max(),
                                                  key="m6")

    blue_avg_level = user_input_col1.number_input("Mavi Takım Seviye Ortalaması", 0,
                                                   9,
                                                   key="m8")

    user_input_col2.subheader("Kırmızı Takım")

    red_kills = user_input_col2.number_input("Öldürme",load_dataset()["redKills"].min(),
                                             load_dataset()["redKills"].max(), key="r1")


    red_assists = user_input_col2.number_input("Asist", load_dataset()["redAssists"].min(),
                                               red_kills * 4, key="r2")

    red_towers_destroyed = user_input_col2.number_input("Yıkılan Kule", load_dataset()["redTowersDestroyed"].min(),
                                                        4, key="r3")
    red_total_minions_killed = user_input_col2.number_input("Toplam Öldürülen Minyon",
                                                            0,
                                                            load_dataset()["redTotalMinionsKilled"].max(), key="r4")
    red_total_jungle_minions_killed = user_input_col2.number_input("Toplam Öldürülen Orman Canavarı", 0,
                                                                   load_dataset()["redTotalJungleMinionsKilled"].max(),
                                                                   key="r5")
    red_total_gold = user_input_col2.number_input("Kırmızı Takım Toplam Altın", 0,
                                                                   load_dataset()["redTotalGold"].max(),
                                                                   key="r6")
    red_avg_level = user_input_col2.number_input("Kırmızı Takım Seviye Ortalaması", 0,
                                                  9,
                                                  key="r8")

    user_input_col3.subheader("Hedefler")
    is_blue_gained_first_blood = user_input_col3.radio("İlk Kanı Kim Aldı?", ["Kırmızı", "Mavi"])
    if is_blue_gained_first_blood == "Mavi":
        is_blue_gained_first_blood = 1
    else:
        is_blue_gained_first_blood = 0
    is_blue_gained_first_dragon = user_input_col3.radio("İlk Ejderhayı Kim Öldürdü?", ["Kırmızı", "Mavi"])
    if is_blue_gained_first_dragon == "Mavi":
        is_blue_gained_first_dragon = 1
    else:
        is_blue_gained_first_dragon = 0


    result_col.subheader("Sonuç: ")

    #"""['blueWins', 'blueWardsPlaced', 'blueWardsDestroyed', 'blueFirstBlood', 'blueKills', 'blueDeaths', 'blueAssists',
     #'blueDragons', 'blueHeralds', 'blueTowersDestroyed', 'blueTotalMinionsKilled', 'blueTotalJungleMinionsKilled',
     #'blueCSPerMin', 'redWardsPlaced', 'redWardsDestroyed', 'redKills', 'redDeaths', 'redAssists', 'redTowersDestroyed',
     #'redTotalMinionsKilled', 'redTotalJungleMinionsKilled', 'redCSPerMin']"""

    user_input = pd.DataFrame({

        "blueFirstBlood": is_blue_gained_first_blood,
        "blueKills": blue_kills,

        "blueAssists": blue_assists,
        "blueDragons": is_blue_gained_first_dragon,
        'blueAvgLevel':blue_avg_level,
        "blueTowersDestroyed": blue_towers_destroyed,
        "blueTotalMinionsKilled": blue_total_minions_killed,
        "blueTotalJungleMinionsKilled": blue_total_jungle_minions_killed,
        "blueTotalGold": blue_total_gold,
        

        "redKills": red_kills,
        'redAvgLevel':red_avg_level,
        "redAssists": red_assists,
        "redTowersDestroyed": red_towers_destroyed,
        "redTotalMinionsKilled": red_total_minions_killed,
        "redTotalJungleMinionsKilled": red_total_jungle_minions_killed,
        "redTotalGold": red_total_gold
    }, index=[0],dtype=int)

    #"""result_col.subheader(user_input.columns)
    #result_col.subheader(user_input.iloc[0,:])"""
    pipeline = load_model()
    if result_col.button("Tahmin Et!"):
        result = pipeline.predict(user_input)[0]
        if result == 1:
            result = "Mavi"
            st.snow()
        else:
            result = "Kırmızı"
            rain(
                emoji="🎈",
                font_size=54,
                falling_speed=5,
                animation_length="10",
            )
        result_col.header(f"{result}", anchor=False)



if __name__ == "__main__":
    main(main_page)
    data(data_page)
    model(model_page)
