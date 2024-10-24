abc = """ what if there are other text here.
and here. 
and here.
<json_list>
[
    {
        "School Name": "Temasek Polytechnic",
        "Course Name": "Diploma in Information Technology",
        "Course Code": "C71",
        "Aggregate Score Range": "8-14",
        "Aggregate Score Type": "ELR2B2-B"
    },
    {
        "Institute Name": "Institute of Technical Education - College Central",
        "Course Name": "Higher Nitec in Information Technology",
        "Course Code": "N43",
        "Aggregate Score Range": "17-33",
        "Aggregate Score Type": "ELR2B2-C"
    },
    {
        "Institute Name": "Nanyang Polytechnic",
        "School Name": "School of Information Technology",
        "Course Name": "Diploma in Information Technology",
        "Course Code": "C71",
        "Aggregate Score Range": "11-16",
        "Aggregate Score Type": "ELR2B2-B"
    },
    {
        "Institute Name": "Singapore Polytechnic",
        "School Name": "School of Computing",
        "Course Name": "Diploma in Information Technology",
        "Course Code": "C67",
        "Aggregate Score Range": "11-16",
        "Aggregate Score Type": "ELR2B2-B"
    }
]
</json_list>here too
and also here"""

import re
import json 
import pandas as pd
df_list=None

json_strings = re.findall(r'<json_list>.+</json_list>', abc, flags = re.DOTALL)
if len(json_strings)>0:
   json_strings = re.sub(r'</?json_list>','', json_strings[0])

try:
    json_objs = json.loads(json_strings)
    df_list = pd.json_normalize(json_objs)
except:
   pass

response_text = re.sub(r'<json_list>.+</json_list>', '', abc, flags = re.DOTALL)


if df_list is not None:
    print(df_list.to_string())
print(response_text)

import streamlit as st
st.dataframe(None)