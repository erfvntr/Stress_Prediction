import pickle
import re
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
from src.exception import CustomException

# налаштування

st.set_page_config(
    page_title="Прогнозування рівня стресу",
    layout="wide"
)

MODEL_PATH = (
    Path(__file__).parent
    / "src"/"notebook_info"/"stress_logistic_model.pkl"
)

# завантаження моделі

@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)
    except Exception as error:
        raise CustomException(error, sys)


try:
    model = load_model()
except CustomException as error:
    st.error("Не вдалося завантажити модель.")
    st.error(str(error))
    st.stop()


# допоміжні функції

def normalize_column_name(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def get_question_type(column):
    name = normalize_column_name(column)

    for scale in ["pss", "gad", "phq"]:
        if re.match(rf"{scale}_?\d+$", name):
            return scale.upper()

    return None


def get_question_code(column):
    # перетворення назв pss1, pss_1 у стандартний формат PSS1
    name = normalize_column_name(column)

    match = re.match(
        r"(pss|gad|phq)_?(\d+)$",
        name
    )

    if match:
        return f"{match.group(1).upper()}{match.group(2)}"

    return str(column)


# запитання психометричних шкал

QUESTION_TEXTS = {

    "PSS1":
        "Як часто протягом семестру ви засмучувалися через події, "
        "пов’язані з вашими навчальними справами?",

    "PSS2":
        "Як часто протягом семестру ви відчували, що не можете "
        "контролювати важливі речі у своєму навчанні?",

    "PSS3":
        "Як часто протягом семестру ви відчували нервозність і стрес "
        "через академічне навантаження?",

    "PSS4":
        "Як часто протягом семестру ви відчували, що не можете "
        "впоратися з усіма обов’язковими навчальними завданнями "
        "(наприклад, завданнями, тестами та іспитами)?",

    "PSS5":
        "Як часто протягом семестру ви були впевнені у своїй здатності "
        "впоратися з навчальними або університетськими проблемами?",

    "PSS6":
        "Як часто протягом семестру ви відчували, що у вашому "
        "навчанні все складається так, як потрібно?",

    "PSS7":
        "Як часто протягом семестру вам вдавалося контролювати "
        "роздратування, пов’язане з навчанням або університетськими справами?",

    "PSS8":
        "Як часто протягом семестру ви відчували, що ваша академічна "
        "успішність перебуває на високому рівні?",

    "PSS9":
        "Як часто протягом семестру ви сердилися через погану успішність "
        "або низькі оцінки, які, на вашу думку, були поза вашим контролем?",

    "PSS10":
        "Як часто протягом семестру ви відчували, що навчальні труднощі "
        "накопичилися настільки, що ви не можете їх подолати?",


    "GAD1":
        "Як часто протягом семестру ви відчували нервозність, тривогу "
        "або напруження через академічний тиск?",

    "GAD2":
        "Як часто протягом семестру ви не могли припинити хвилюватися "
        "через свої навчальні справи?",

    "GAD3":
        "Як часто протягом семестру вам було складно розслабитися "
        "через академічний тиск?",

    "GAD4":
        "Як часто протягом семестру ви легко дратувалися "
        "через академічний тиск?",

    "GAD5":
        "Як часто протягом семестру ви надмірно хвилювалися "
        "через свої навчальні справи?",

    "GAD6":
        "Як часто протягом семестру через академічний тиск ви відчували "
        "настільки сильне занепокоєння, що вам було складно всидіти на місці?",

    "GAD7":
        "Як часто протягом семестру ви відчували страх, "
        "ніби може статися щось погане?",

    "PHQ1":
        "Як часто протягом семестру ви відчували слабкий інтерес "
        "або задоволення від повсякденних справ?",

    "PHQ2":
        "Як часто протягом семестру ви відчували пригнічений "
        "або безнадійний настрій?",

    "PHQ3":
        "Як часто протягом семестру у вас виникали труднощі із засинанням, "
        "підтриманням сну або ви спали занадто багато?",

    "PHQ4":
        "Як часто протягом семестру ви відчували втому "
        "або нестачу енергії?",

    "PHQ5":
        "Як часто протягом семестру у вас був поганий апетит "
        "або схильність до переїдання?",

    "PHQ6":
        "Як часто протягом семестру ви були незадоволені собою "
        "або відчували, що підвели себе чи свою родину?",

    "PHQ7":
        "Як часто протягом семестру вам було складно зосередитися, "
        "наприклад під час читання або перегляду телевізора?",

    "PHQ8":
        "Як часто протягом семестру ви помічали значне уповільнення "
        "рухів чи мовлення або, навпаки, незвичну неспокійність?",

    "PHQ9":
        "Як часто протягом семестру у вас виникали небезпечні думки "
        "щодо власного життя або безпеки?"
}


# варіанти відповідей

ANSWER_OPTIONS = {

    "PSS": {
        0: "Ніколи",
        1: "Майже ніколи",
        2: "Іноді",
        3: "Досить часто",
        4: "Дуже часто"
    },

    "GAD": {
        0: "Зовсім ні",
        1: "Кілька днів",
        2: "Більше половини днів",
        3: "Майже щодня"
    },

    "PHQ": {
        0: "Зовсім ні",
        1: "Кілька днів",
        2: "Більше половини днів",
        3: "Майже щодня"
    }
}


def render_scale(features, title, scale_type, expanded=False):
    # відображення запитань та отримання числових відповідей
    values = {}

    if not features:
        return values

    st.divider()
    st.header(title)

    answers = ANSWER_OPTIONS[scale_type]

    with st.expander(
        "Заповнити відповіді",
        expanded=expanded
    ):
        for feature in features:

            question_code = get_question_code(feature)

            question = QUESTION_TEXTS.get(
                question_code,
                feature
            )

            values[feature] = st.selectbox(
                question,
                options=list(answers.keys()),
                format_func=lambda value, options=answers:
                    f"{value} — {options[value]}",
                key=f"input_{feature}"
            )

    return values


# назви класів прогнозу

CLASS_NAMES = {
    0: "Низький рівень стресу",
    1: "Помірний рівень стресу",
    2: "Високий рівень сприйнятого стресу",
    "Low Stress": "Низький рівень стресу",
    "Moderate Stress": "Помірний рівень стресу",
    "High Perceived Stress": "Високий рівень сприйнятого стресу"
}


# категоріальні ознаки

CATEGORY_OPTIONS = {

    "age": [
        "18-22",
        "23-26",
        "27-30"
    ],

    "gender": [
        "Male",
        "Female"
    ],

    "department": [
        "CS Engineering",
        "Business / Entrepreneurship",
        "EEE/ECE Engineering",
        "Biological Sciences",
        "Mechanical Engineering",
        "Civil Engineering",
        "Env/Life Sciences",
        "Other"
    ],

    "academic_year": [
        "First",
        "Second",
        "Third",
        "Forth",
        "Other"
    ],

    "year": [
        "First",
        "Second",
        "Third",
        "Forth",
        "Other"
    ],

    "cgpa": [
        "Below 2.50",
        "2.50-2.99",
        "3.00-3.39",
        "3.40-3.79",
        "3.80-4.00",
        "Other"
    ],

    "scholarship_waiver": [
        "No",
        "Yes"
    ],

    "scholarship": [
        "No",
        "Yes"
    ]
}


CATEGORY_LABELS = {
    "age": "Вік",
    "gender": "Стать",
    "department": "Факультет / спеціальність",
    "academic_year": "Курс навчання",
    "year": "Курс навчання",
    "cgpa": "Середній академічний бал (CGPA)",
    "scholarship_waiver": "Наявність стипендії / знижки",
    "scholarship": "Наявність стипендії / знижки"
}


# визначення ознак, які очікує модель

if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)

else:
    expected_features = [
        "age",
        "gender",
        "department",
        "academic_year",
        "cgpa",
        "scholarship_waiver"
    ]

    expected_features += [
        f"PSS{i}" for i in range(1, 11)
    ]

    expected_features += [
        f"GAD{i}" for i in range(1, 8)
    ]

    expected_features += [
        f"PHQ{i}" for i in range(1, 10)
    ]


# розподіл ознак за шкалами

pss_features = [
    feature
    for feature in expected_features
    if get_question_type(feature) == "PSS"
]

gad_features = [
    feature
    for feature in expected_features
    if get_question_type(feature) == "GAD"
]

phq_features = [
    feature
    for feature in expected_features
    if get_question_type(feature) == "PHQ"
]


# інтерфейс застосунку

st.title(
    "Прогнозування рівня стресу студентів"
)

st.write(
    """
    Вебзастосунок використовує модель **логістичної регресії**
    для прогнозування рівня стресу студента на основі
    соціально-демографічних даних та відповідей на запитання
    психометричних шкал.
    """
)

st.info(
    "Результат має інформаційний характер і не є "
    "медичним або психологічним діагнозом."
)


# форма введення даних

with st.form("stress_prediction_form"):

    input_data = {}

    st.header("Загальна інформація")

    col1, col2 = st.columns(2)

    category_counter = 0

    # введення категоріальних параметрів

    for feature in expected_features:

        normalized = normalize_column_name(feature)

        if normalized not in CATEGORY_OPTIONS:
            continue

        column = (
            col1
            if category_counter % 2 == 0
            else col2
        )

        with column:

            input_data[feature] = st.selectbox(
                CATEGORY_LABELS.get(
                    normalized,
                    feature
                ),
                CATEGORY_OPTIONS[normalized],
                key=f"category_{feature}"
            )

        category_counter += 1


    # введення відповідей pss-10

    input_data.update(
        render_scale(
            pss_features,
            "Шкала сприйнятого стресу — PSS-10",
            scale_type="PSS",
            expanded=True
        )
    )


    # введення відповідей gad-7

    input_data.update(
        render_scale(
            gad_features,
            "Шкала тривожності — GAD-7",
            scale_type="GAD"
        )
    )


    # введення відповідей phq-9

    input_data.update(
        render_scale(
            phq_features,
            "Шкала депресивних симптомів — PHQ-9",
            scale_type="PHQ"
        )
    )


    # визначення інших ознак

    known_features = (
        set(pss_features)
        | set(gad_features)
        | set(phq_features)
    )

    unknown_features = [
        feature
        for feature in expected_features
        if (
            normalize_column_name(feature)
            not in CATEGORY_OPTIONS
            and feature not in known_features
        )
    ]

    if unknown_features:

        st.divider()
        st.header("Інші параметри")

        for feature in unknown_features:

            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                key=f"other_{feature}"
            )

    st.divider()

    submitted = st.form_submit_button(
        "Визначити рівень стресу",
        use_container_width=True
    )


# прогнозування

if submitted:

    try:

        # формування dataframe з введених даних

        input_df = pd.DataFrame(
            [input_data]
        )

        # встановлення порядку колонок як при навчанні моделі

        input_df = input_df.reindex(
            columns=expected_features
        )

        # отримання прогнозу

        prediction = model.predict(
            input_df
        )[0]

        prediction_name = CLASS_NAMES.get(
            prediction,
            str(prediction)
        )

        # виведення результату

        st.divider()
        st.header(
            "Результат прогнозування"
        )

        if prediction in [
            0,
            "Low Stress"
        ]:

            st.success(
                f"Прогнозований рівень: "
                f"**{prediction_name}**"
            )

        elif prediction in [
            1,
            "Moderate Stress"
        ]:

            st.warning(
                f"Прогнозований рівень: "
                f"**{prediction_name}**"
            )

        elif prediction in [
            2,
            "High Perceived Stress"
        ]:

            st.error(
                f"Прогнозований рівень: "
                f"**{prediction_name}**"
            )

        else:

            st.info(
                f"Прогнозований клас: "
                f"**{prediction_name}**"
            )

    # обробка помилок

    except Exception as error:

        custom_error = CustomException(
            error,
            sys
        )

        st.error(
            "Не вдалося виконати прогноз."
        )

        st.error(
            str(custom_error)
        )