import streamlit as st
import pandas as pd
import pickle
import os

# পেজের টাইটেল ও ইন্টারফেস সুন্দর করা
st.set_page_config(page_title="Student Math Score Predictor AI", page_icon="🎓")
st.title("🎓 Student Performance Predictor AI Agent")
st.write("শিক্ষার্থীর রিডিং ও রাইটিং স্কোর দিন, এআই এজেন্ট তার সম্ভাব্য গণিত নম্বর প্রেডিক্ট করে দেবে।")

# ইনপুট নেওয়ার জন্য স্লাইডার বা বক্স তৈরি
reading_score = st.number_input("Reading Score (০ থেকে ১০০)", min_value=0, max_value=100, value=75)
writing_score = st.number_input("Writing Score (০ থেকে ১০০)", min_value=0, max_value=100, value=78)

# প্রেডিকশন বাটন
if st.button("Predict Math Score"):
    # ফাইলের নাম ছোট হাতের বা বড় হাতের যাই হোক না কেন তা খুঁজে বের করার লজিক
    model_file = None
    for file in os.listdir('.'):
        if file.lower() == 'student_model.pkl':
            model_file = file
            break
            
    if model_file and os.path.exists(model_file):
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
        
        # ইনপুট ডেটা ফরম্যাট করা
        input_data = pd.DataFrame([[reading_score, writing_score]], columns=['reading score', 'writing score'])
        
        # প্রেডিকশন
        prediction = model.predict(input_data)[0]
        
        # স্ক্রিনে সুন্দর করে রেজাল্ট দেখানো
        st.success(f"🎯 এআই এজেন্টের মতে শিক্ষার্থীর সম্ভাব্য গণিত নম্বর হবে: **{prediction:.2f}**")
    else:
        st.error("মডেল ফাইলটি (student_model.pkl) খুঁজে পাওয়া যায়নি। দয়া করে গিটহাবে ফাইলটি আপলোড করা আছে কিনা চেক করুন।")
