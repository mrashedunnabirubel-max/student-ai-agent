import io
import pandas as pd
import pickle
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Student Math Score Predictor AI", page_icon="🎓")
st.title("🎓 Student Performance Predictor AI Agent")
st.write(
    "শিক্ষার্থীর রিডিং ও রাইটিং স্কোর দিন, এআই এজেন্ট তার সম্ভাব্য গণিত নম্বর প্রেডিক্ট করে দেবে।"
)


# আসল ডেটাসেট সরাসরি কোডের ভেতরে টেক্সট আকারে (কোনো এক্সটার্নাল লিঙ্কের ঝামেলা নেই)
@st.cache_resource
def get_static_model():
    # তোমার আসল ডেটাসেটের প্রথম কয়েকশো রো এর প্যাটার্ন ডেটা
    csv_data = """reading score,writing score,math score
72,74,72
90,88,69
95,93,90
57,44,47
78,75,76
83,78,71
88,92,88
43,39,40
64,67,64
60,50,38
54,52,58
52,43,40
81,73,65
72,74,78
53,58,50
75,78,69
89,86,88
67,28,18
42,46,46
58,61,54
69,63,58
70,70,65
53,53,44
73,73,69
75,80,74
74,72,73
54,55,33
69,63,67
70,65,70
70,61,62
64,74,69
51,54,63
72,65,56
24,23,28
41,41,30
41,37,45
74,74,79
50,51,50
75,76,75
56,57,57
64,68,55
50,43,58
58,65,53
59,65,59
56,54,50
26,10,29
52,51,55
81,81,66
32,38,57
83,84,82
77,74,53
55,61,77
44,51,44
78,75,88
51,54,71
40,38,33
60,63,46
43,41,52
59,58,58
55,49,0
64,56,44
58,48,39
62,59,62
59,63,69
59,58,59
43,43,41
74,70,45
72,74,60
43,45,61
44,50,40
51,46,50
55,63,56
72,75,41
49,56,49
75,78,49
26,22,44
57,55,40
40,48,49
80,72,61
44,43,62
47,42,41
81,78,49
84,82,71
44,48,46
66,71,86
73,70,71
62,74,68
62,63,81
70,67,58
78,75,73
80,82,79
42,38,39
89,86,71
43,45,43
92,92,79
88,85,94
70,70,65
72,76,63
63,62,58
67,67,65
"""
    df = pd.read_csv(io.StringIO(csv_data))
    X = df[["reading score", "writing score"]]
    y = df["math score"]

    model = LinearRegression()
    model.fit(X, y)
    return model


# অ্যাপ শুরুতেই মডেল ট্রেইন করে ফেলবে
model = get_static_model()

# ইউজার ইন্টারফেস
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

    # নম্বর যাতে ০ থেকে ১০০ এর বাইরে না যায় তার বাউন্ডারি সেট করা
    prediction = max(0, min(100, prediction))

    st.success(
        f"🎯 এআই এজেন্টের মতে শিক্ষার্থীর সম্ভাব্য গণিত নম্বর হবে: **{prediction:.2f}**"
    )
