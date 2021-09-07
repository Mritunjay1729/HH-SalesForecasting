# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 00:22:19 2021

@author: MRITYUNJAY
"""
import pandas as pd
import numpy as np

#1st Sheet CDCW
cdcw = pd.read_excel('Data Sheet for Interns.xlsx', sheet_name=0, header = 1)
cdcw.dropna(inplace = True)
cdcw.columns
index = cdcw[cdcw["Month"] == 'Month'].index
cdcw = cdcw.drop(index)
search = ["Carry forwarded", "Carry Forward","carry forward", "Fees Received", "Corporate Batch", "CARRY FORW"]
loc = []
for i in search:
    loc.append(cdcw[cdcw["Fees Received"] == i].index)
for i in loc:
    cdcw = cdcw.drop(i)
cdcw.reset_index()
cdcw["Fees Total"].dtype
cdcw[["Fees Total", "Fees Received", "Fees Pending"]] = cdcw[["Fees Total", "Fees Received", "Fees Pending"]].astype('float')
cdcw.to_excel('cdcw.xlsx')
cdcw


#2nd Sheet  GST SSBB
gst = pd.read_excel('Data Sheet for Interns.xlsx', sheet_name=1, header = 1)
gst.columns
gst['Course Name'].fillna("GST", inplace = True)
gst.dropna(inplace = True)
gst
search = gst[gst["Month"] == 'Month'].index
gst = gst.drop(search)
gst.reset_index()
gst.to_excel('gst.xlsx')
gst = pd.read_excel('gst.xlsx', index_col = 0)
gst.columns
gst[["Fees Total", "Fees Received", "Fees Pending"]] = gst[["Fees Total", "Fees Received", "Fees Pending"]].astype('float')
gst.to_excel('gst.xlsx')
gst.columns

#3rd Sheet
SSBB = pd.read_excel('Data Sheet for Interns.xlsx', sheet_name = 2, header = 1)
SSBB.dropna(inplace = True)
SSBB.columns
SSBB.rename(columns = {"Unnamed: 0":'Month'}, inplace = True)
SSBB.drop('index', axis = 1, inplace = True)
search = SSBB[SSBB["Fees Total"] == "Fees Total"].index
SSBB = SSBB.drop(search)
SSBB.select_dtypes(include = 'object')
search = SSBB[SSBB["Councellor Name"] == 'Consultant'].index
search = SSBB[SSBB["Councellor Name"] == "Consultant"].index
SSBB.drop(search, inplace = True)
SSBB.reset_index(inplace = True)
SSBB.to_excel('SSBB.xlsx')
SSBB.drop('index', axis = 1, inplace = True)
SSBB = pd.read_excel("SSBB.xlsx", index_col=0)
SSBB["Fees Pending"] = SSBB["Fees Total"].astype('float') - SSBB["Fees Received"].astype('float')
SSBB.columns
SSBB.describe()
SSBB.to_excel('SSBB.xlsx')

#4th Sheet SSGB
SSGB = pd.read_excel('Data Sheet for Interns.xlsx', sheet_name = 3, header = 1)
SSGB.rename(columns = {"Unnamed: 0" :"Month"}, inplace = True)
SSGB.dropna(inplace = True)
search = SSGB[SSGB["Councellor Name"] == "Consultant"].index
SSGB.drop(search, inplace = True)
SSGB.reset_index(inplace = True)
SSGB.drop("index", axis = 1, inplace = True)
SSGB = pd.read_excel('SSGB.xlsx')
SSGB["Fees Pending"] = SSGB["Fees Total"].astype('float') - SSGB["Fees Received"].astype('float')
SSGB.describe()
SSGB.to_excel("SSGB.xlsx")

#5th Sheet Analytics
Analytics = pd.read_excel("Data Sheet for Interns.xlsx", sheet_name = 4, header = 1)
Analytics.dropna(inplace= True)
Analytics.rename(columns={"2021-03-01 00:00:00": "Month"}, inplace = True)
Analytics
search = Analytics[Analytics["Councellor Name"] == "Consultant"].index
Analytics.drop(search, inplace = True)
search = Analytics[Analytics["Councellor Name"] == "Councellor Name"].index
Analytics.drop(search, inplace = True)
Analytics.reset_index(inplace = True)
Analytics.drop("index", axis = 1, inplace = True)
Analytics.to_excel("Analytics.xlsx")
Analytics

#6th Sheet PMP
PMP = pd.read_excel("Data Sheet for Interns.xlsx", sheet_name = 5, header = 1)
PMP.rename(columns={"2020-11-01 00:00:00": "Month"}, inplace = True)
PMP.dropna(inplace = True)
search = PMP[PMP["Councellor Name"] == "Consultant"].index
PMP.drop(search, inplace = True)
search = PMP[PMP["Councellor Name"] == "Councellor Name"].index
PMP.drop(search, inplace = True)
PMP.reset_index(drop = True, inplace = True)
PMP.to_excel("PMP.xlsx")
PMP = pd.read_excel('PMP.xlsx')
PMP.columns
PMP["Fees Pending"] = PMP["Fees Total"].astype('float') - PMP["Fees Received"].astype('float')
PMP.drop("Unnamed: 0", inplace = True, axis = 1)


#Combined hufffffffff Finally Done with it
PMP = pd.read_excel('PMP.xlsx', index_col = 0)
Analytics = pd.read_excel("Analytics.xlsx", index_col = 0)
SSGB = pd.read_excel('SSGB.xlsx', index_col = 0)
gst = pd.read_excel("gst.xlsx", index_col = 0)
cdcw = pd.read_excel('cdcw.xlsx', index_col = 0)
SSBB = pd.read_excel("SSBB.xlsx", index_col = 0)

PMP.columns
Analytics.shape
SSGB.shape
gst.shape
cdcw.shape
SSBB.shape
combined_sales.shape
combined = [PMP, Analytics, SSGB, gst, cdcw, SSBB]
combined_sales = pd.concat(combined, axis = 0)
combined_sales.reset_index(drop=True, inplace = True)
combined_sales.to_excel("Combined Sales.xlsx")
combined_sales.tail(100)
