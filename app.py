import os
import pandas as pd
import pickle
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Student Math Score Predictor AI", page_icon="🎓")
st.title("🎓 Student Performance Predictor AI Agent")
st.write(
    "শিক্ষার্থীর রিডিং ও রাইটিং স্কোর দিন, এআই এজেন্ট তার সম্ভাব্য গণিত নম্বর প্রেডিক্ট করে দেবে।"
)


# ব্যাকগ্রাউন্ডে অটোমেটিক মডেল তৈরি করার ফাংশন (কোনো আলাদা ফাইলের দরকার নেই)
@st.cache_resource
def train_model_live():
    # সরাসরি অনলাইন স্টোরেজ থেকে তোমার আসল ডেটাসেট লোড করা হচ্ছে
    url = "https://raw.githubusercontent.com/srinivasav22/Machine-Learning-Program/main/StudentsPerformance.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.lower().str.strip()

    reading_col = [c for c in df.columns if "reading" in c][0]
    writing_col = [c for c in df.columns if "writing" in c][0]
    math_col = [c for c in df.columns if "math" in c][0]

    X = df[[reading_col, writing_col]]
    y = df[math_col]

    model = LinearRegression()
    model.fit(X, y)
    return model


# মডেল লোড করা
try:
    model = train_model_live()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(
        "ইন্টারনেট সমস্যার কারণে ব্যাকগ্রাউন্ডে ডেটা লোড করা যাচ্ছে না। দয়া করে পেজটি রিফ্রেশ করুন।"
    )

# ইনপুট ও প্রেডিকশন লজিক
if model_loaded:
    reading_score = st.number_input(
        "Reading Score (০ থেকে ১০০)", min_value=0, max_value=100, value=75
    )
    writing_score = st.number_input(
        "Writing Score (০ থেকে ১০০)", min_value=0, max_value=100, value=78
    )

    if st.button("Predict Math Score"):
        input_data = pd.DataFrame(
            [[reading_score, writing_score]],
            columns=["reading score", "writing score"],
        )
        prediction = model.predict(input_data)[0]
        st.success(
            f"🎯 এআই এজেন্টের মতে শিক্ষার্থীর সম্ভাব্য গণিত নম্বর হবে: **{prediction:.2f}**"
        )
