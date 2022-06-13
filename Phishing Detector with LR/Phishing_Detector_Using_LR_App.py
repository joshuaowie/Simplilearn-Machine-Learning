#!/usr/bin/env python
# coding: utf-8
# In[26]:


import pandas as pd
import numpy as np
import streamlit as st
import pickle


# In[27]:
st.write("""
# Phishing Detector Using LR App

This app predicts from the input features of a site whether it is a phishing site or not
""")
st.write('---')

phishing = pd.read_csv(r'phishing.txt')


# In[28]:


phishing.shape


# In[29]:


phishing.columns = ['UsingIP', 'LongURL', 'ShortURL', 'Symbol@', 'Redirecting//',
                    'PrefixSuffix-', 'SubDomains', 'HTTPS', 'DomainRegLen', 'Favicon',
                    'NonStdPort', 'HTTPSDomainURL', 'RequestURL', 'AnchorURL',
                    'LinksInScriptTags', 'ServerFormHandler', 'InfoEmail', 'AbnormalURL',
                    'WebsiteForwarding', 'StatusBarCust', 'DisableRightClick',
                    'UsingPopupWindow', 'IframeRedirection', 'AgeofDomain',
                    'DNSRecording', 'WebsiteTraffic', 'PageRank', 'GoogleIndex',
                    'LinksPointingToPage', 'StatsReport', 'class']


# In[30]:
st.sidebar.write("""
# Specify Input for site features

Each input feature is part of the characteristics for a site
""")
st.write('---')


def user_input():
    UsingIP = st.sidebar.selectbox('UsingIP', phishing['UsingIP'].unique())
    LongURL = st.sidebar.selectbox('LongURL', phishing['LongURL'].unique())
    ShortURL = st.sidebar.selectbox('ShorURL', phishing['ShortURL'].unique())
    Symbol = st.sidebar.selectbox('Symbol@', phishing['Symbol@'].unique())
    Redirecting = st.sidebar.selectbox(
        'Redirecting', phishing['Redirecting//'].unique())
    PrefixSuffix = st.sidebar.selectbox(
        'PreffixSuffix-', phishing['PrefixSuffix-'].unique())
    SubDomains = st.sidebar.selectbox(
        'SubDomains', phishing['SubDomains'].unique())
    HTTPS = st.sidebar.selectbox('HTTPS', phishing['HTTPS'].unique())
    DomainRegLen = st.sidebar.selectbox(
        'DomainRegLen', phishing['DomainRegLen'].unique())
    Favicon = st.sidebar.selectbox('Favicon', phishing['Favicon'].unique())
    NonStdPort = st.sidebar.selectbox(
        'NonStdPort', phishing['NonStdPort'].unique())
    HTTPSDomainURL = st.sidebar.selectbox(
        'HTTPS Domain URL', phishing['HTTPSDomainURL'].unique())
    RequestURL = st.sidebar.selectbox(
        'Request URL', phishing['RequestURL'].unique())
    AnchorURL = st.sidebar.selectbox(
        'Anchor URL', phishing['AnchorURL'].unique())
    LinksInScriptTags = st.sidebar.selectbox(
        'Links in Script Tags', phishing['LinksInScriptTags'].unique())
    ServerFormHandler = st.sidebar.selectbox(
        'Server Form Handler', phishing['ServerFormHandler'].unique())
    InfoEmail = st.sidebar.selectbox(
        'InfoEmail', phishing['InfoEmail'].unique())
    AbnormalURL = st.sidebar.selectbox(
        'AbnormalURL', phishing['AbnormalURL'].unique())
    WebsiteForwarding = st.sidebar.selectbox(
        'Website Forwarding', phishing['WebsiteForwarding'].unique())
    StatusBarCust = st.sidebar.selectbox(
        'Status Bar Cust', phishing['StatusBarCust'].unique())
    DisableRightClick = st.sidebar.selectbox(
        'Disable Right Click', phishing['DisableRightClick'].unique())
    UsingPopupWindow = st.sidebar.selectbox(
        'Using Popup Window', phishing['UsingPopupWindow'].unique())
    IframeRedirection = st.sidebar.selectbox(
        'Iframe Redirection', phishing['IframeRedirection'].unique())
    AgeofDomain = st.sidebar.selectbox(
        'Age Of Domain', phishing['AgeofDomain'].unique())
    DNSRecording = st.sidebar.selectbox(
        'DNS Recording', phishing['DNSRecording'].unique())
    WebsiteTraffic = st.sidebar.selectbox(
        'Website Traffick', phishing['WebsiteTraffic'].unique())
    PageRank = st.sidebar.selectbox(
        'Page Rank', phishing['GoogleIndex'].unique())
    GoogleIndex = st.sidebar.selectbox(
        'Google Index', phishing['GoogleIndex'].unique())
    LinksPointingToPage = st.sidebar.selectbox(
        'Links Pointing To Page', phishing['LinksPointingToPage'].unique())
    StatsReport = st.sidebar.selectbox(
        'Stats Report', phishing['StatsReport'].unique())
    Data = {
        'UsingIP': UsingIP,
        'LongURL': LongURL,
        'ShortURL': ShortURL,
        'Symbol@': Symbol,
        'Redirecting//': Redirecting,
        'PrefixSuffix-': PrefixSuffix,
        'SubDomains': SubDomains,
        'HTTPS': HTTPS,
        'DomainRegLen': DomainRegLen,
        'Favicon': Favicon,
        'NonStdPort': NonStdPort,
        'HTTPSDomainURL': HTTPSDomainURL,
        'RequestURL': RequestURL,
        'AnchorURL': AnchorURL,
        'LinksInScriptTags': LinksInScriptTags,
        'ServerFormHandler': ServerFormHandler,
        'InfoEmail': InfoEmail,
        'AbnormalURL': AbnormalURL,
        'WebsiteForwarding': WebsiteForwarding,
        'StatusBarCust': StatusBarCust,
        'DisableRightClick': DisableRightClick,
        'UsingPopupWindow': UsingPopupWindow,
        'IframeRedirection': IframeRedirection,
        'AgeofDomain': AgeofDomain,
        'DNSRecording': DNSRecording,
        'WebsiteTraffic': WebsiteTraffic,
        'PageRank': PageRank,
        'GoogleIndex': GoogleIndex,
        'LinksPointingToPage': LinksPointingToPage,
        'StatsReport': StatsReport,
    }
    features = pd.DataFrame(Data, index=[0])
    return features


input_df = user_input()

# add user input to the trained dataset of previous sites
phishing_df = phishing.drop('class', axis=1)
phishing_df = pd.concat([input_df, phishing_df], axis=0)

# select only the user input from the top of the dataset
phishing_df_input = phishing_df[:1]

# display user input details
st.header('User Input Features')
st.write(phishing_df_input)

# load the previously saved classifier model file
load_clf = pickle.load(open('phishing_detector_rf.pkl', 'rb'))

# apply model to make prediction
prediction = load_clf.predict(phishing_df_input)
prediction_proba = load_clf.predict_proba(phishing_df_input)


st.subheader('Prediction where phishing website(1) or not(-1)')

#phishing_site = np.array(['Phishing Website', 'Not Phishing Website'])
st.write(prediction)


st.subheader('Prediction Probability')
st.write(prediction_proba)

# %%
