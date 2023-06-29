#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from bs4 import BeautifulSoup
import requests as rs
import re
import time
import random
import pandas as pd
import numpy as np
from selenium import webdriver
import chromedriver_autoinstaller
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_auto_update import check_driver
links_dont_work=[]
appartaments=[]
slep_arr=[2,3,5,7]
list_items=['לשוטפים','ריהוט','מעלית','מזגן','חניה','מרפסת','סורגים','ממד','מחסן','משופצת','דוד שמש','חיות מחמד']
def openSp(link_name):
    x = rs.get(link_name)
    soup=BeautifulSoup(x.content,'html.parser')
    print(x.status_code)
    return soup
def Lists_app(link_into):
    serv = Service(ChromeDriverManager().install())
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(serv.path)
    driver.get(link_into)
    ps = driver.page_source
    soup = BeautifulSoup(ps, 'html.parser')

def inside_app(appartament,link):
    soup=openSp(link)
    divs=soup.find_all('div',attrs={'class':'row'})
    count=1
    link_checks=soup.find_all('div',attrs={'class':'card-icons-wrap p-3'})
    for d in divs:
        h2_tit=d.find_all('h2',attrs={'class':'card-title'})
        exep=d.find_all('p',attrs={'class':'text-word-break'})
        for i in range(0,len(exep)):
            if i==0:
                appartament['תאור']=exep[0].text
            else:
                break
        tables=d.find_all('table',attrs={'class':'table table-sm mb-4'})
        my_tr=d.find_all('td')
        for m in range(0,len(my_tr)):
            my_sp = my_tr[m].text
            pattern = r'\d{2}/\d{2}/\d{4}'
            match = re.search(pattern, my_sp)
            if maych:
                year = my_sp.split('/')[-1]
                mount= my_sp.split('/')[-2]
                if year<2023 and mount<6:
                    break
            if m%2!=0:
                if m == 13:
                    split_text = re.split(r'\s+', my_sp)
                    appartament['קומה'] = split_text[1]
                else:
                    if m==1:
                        clean_text = my_tr[m].text.strip()
                        appartament['סוג הנכס']=clean_text
                    if m==3:
                        clean_text = my_tr[m].text.strip()
                        appartament['עיר'] = clean_text
                    if m==5:
                        clean_text = my_tr[m].text.strip()
                        appartament['שכונה'] = clean_text
                    if m==7:
                        clean_text = my_tr[m].text.strip()
                        appartament['רחוב'] = clean_text
                    if m==9:
                        clean_text = my_tr[m].text.strip()
                        appartament['חדרים'] = clean_text
                    if m==15:
                        clean_text = my_tr[m].text.strip()
                        appartament['מטרים'] = clean_text
        for el in h2_tit:
            if count==2:
                  appartament['מחיר'] = el.text
            count+=1
    list_items_in_ap = ['ריהוט', 'מעלית', 'מזגן', 'חניה', 'מרפסת', 'סורגים', 'ממ"ד', 'מחסן', 'משופצת']
    list_items_ap = ['ריהוט', 'מעלית', 'מזגן', 'חניה', 'מרפסת', 'סורגים', 'ממד', 'מחסן', 'משופצת']
    for lk in link_checks:
        my_a=lk.find_all('i')
        span=lk.find_all('span',attrs={'class':'px-1'})
        for i in range(0,len(my_a)):
            last=my_a[i].get('class')
            text=span[i].text
            for j in range(0,len(list_items_in_ap)):
                if re.search(list_items_in_ap[j],text):
                    if re.search('check',last[2]):
                        appartament[list_items_ap[j]] = True
            
    time.sleep(slep_arr[random.randint(0, 3)])
    return  appartament

def ad_info():
    appartaments2 = []
    url="https://www.ad.co.il/nadlanrent?pageindex="
    lk="https://www.ad.co.il"
    count=1
    for page_count in range(0,300):
        my_st=str(page_count)
        my_soup = openSp(url+''+my_st)
        total_div = my_soup.find_all('div', attrs={'class': 'cards-wrap s m l'})
        for t in total_div:
            lit_div=t.find_all('div',attrs={'class':'card overflow-hidden'})
            for ld in lit_div:
                appartament = {
                    'סוג הנכס': '',
                    'עיר': '',
                    'שכונה': '',
                    'רחוב': '',
                    'חדרים': '',
                    'קומה': '',
                    'מחיר': '',
                    'מטרים': '',
                    'לשוטפים': False,
                    'ריהוט': False,
                    'מעלית': False,
                    'מזגן': False,
                    'חניה': False,
                    'מרפסת': False,
                    'סורגים': False,
                    'ממד': False,
                    'מחסן': False,
                    'משופצת': False,
                    'דוד שמש': False,
                    'חיות מחמד': False,
                    'תאור': ''
                }
                ref=ld.find('a').get('href')
                appartament=inside_app(appartament,lk+''+ref)
                appartaments.append(appartament)
                ap_count+=1
        page_count+=1
        df = pd.DataFrame(appartaments)
        df.to_csv('apartaments_ad.csv')
        time.sleep(slep_arr[random.randint(0, 3)])
    count+=1
    print(count)
ad_info()
df=pd.read_csv('apartaments.csv')


# In[ ]:



###[{(def info_inside_homless(appartament,link_into):
    print(link_into)
    serv = Service(ChromeDriverManager().install())
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(serv.path)
    driver.get(link_into)
    ps = driver.page_source
    soup = BeautifulSoup(ps, 'html.parser')
    div=soup.find('div',attrs={'class':'remarks'})
    find_metr=soup.find_all('div',attrs={'id':'ctl00_ContentPlaceHolder1_MainDetails_AdPanel'})
    descr=div.find('span').get_text()
    appartament['תאור']=descr
    list_check=soup.find_all('img',attrs={'class':'itemsAd'})
    for d in find_metr:
        st=d.text
        text=st.split('\n')
        for t in text:
            if re.search('מ"ר:',t):
                t=t.split(' ')
                appartament['מטרים'] =t[1]
                break
    place=0
    for pic in list_check:
        socre_img=pic.get('src')
        if re.search('checked',socre_img):
             appartament[list_items[place]]=True
        place+=1
    driver.quit() 
    print(appartament)
    time.sleep(slep_arr[random.randint(0,3)])
    return  appartament
def get_info_of_app_homless(link_name):
    for next_page in range(1,101):
        serv = Service(ChromeDriverManager().install())
        options = Options()
        options.headless = True
        driver = webdriver.Chrome(serv.path)
        page = "% s" % next_page
        driver.get(link_name+''+page)
        ps = driver.page_source
        soup = BeautifulSoup(ps, 'html.parser')
        aps_info=soup('tr',attrs={'class':'light'})
        print(len(aps_info))
        for ap in aps_info:
            appartament = {
                'סוג הנכס': '',
                'עיר': '',
                'שכונה': '',
                'רחוב': '',
                'חדרים': '',
                'קומה': '',
                'מחיר': '',
                'מטרים': '',
                'לשוטפים': False,
                'ריהוט': False,
                'מעלית': False,
                'מזגן': False,
                'חניה': False,
                'מרפסת': False,
                'סורגים': False,
                'ממד': False,
                'מחסן': False,
                'משופצת': False,
                'דוד שמש': False,
                'חיות מחמד': False,
                'תאור': ''
            }
            td_list=ap.find_all('td')
            link_ap = ap.find('a').get('href')
            count = 0
            for td in td_list:
                flag=1
                str=td.get_text()
                if count>0 and count<9:
                    if count==2:
                        appartament['סוג הנכס']=str
                    if count==3:
                        appartament['עיר'] = str
                    if count==4:
                        appartament['שכונה'] = str
                    if count==5:
                        appartament['רחוב'] = str
                    if count==6:
                        appartament['חדרים'] = str
                    if count==7:
                        appartament['קומה'] = str
                    if count==8:
                        appartament['מחיר']=str
                    if count==9:
                        my_sp=str
                        pattern = r'\d{2}/\d{2}/\d{4}'
                        match = re.search(pattern, my_sp)
                        if maych:
                            year = my_sp.split('/')[-1]
                            mount= my_sp.split('/')[-2]
                            if year<2023 and mount<6:
                                flag=0
                count+=1
            if flag:
                try:
                    appartament=info_inside_homless(appartament,link_name+''+link_ap)
                    links_dont_work.append(link_name + '' + link_ap)
                    appartaments.append(appartament)
                except:
                    print('not work\n'+link_name+''+link_ap)

        driver.quit()  
        df=pd.DataFrame(appartaments)
        df.to_csv('apartaments.csv')
        print(next_page)
        time.sleep(slep_arr[random.randint(0,3)])

appartament ={
                'סוג הנכס': '',
                'עיר': '',
                'שכונה': '',
                'רחוב': '',
                'חדרים': '',
                'קומה': '',
                'מחיר': '',
                'מטרים': '',
                'לשוטפים': False,
                'ריהוט': False,
                'מעלית': False,
                'מזגן': False,
                'חניה': False,
                'מרפסת': False,
                'סורגים': False,
                'ממד': False,
                'מחסן': False,
                'משופצת': False,
                'דוד שמש': False,
                'חיות מחמד': False,
                'תאור': ''
            }
get_info_of_app_homless('https://www.homeless.co.il/rent/')}]


# In[ ]:


import requests as rs
import re
import time
import random
import pandas as pd
import numpy as np
def getData(file_name):
    df=pd.read_csv(file_name)
    return  df

def del_simbols_from_price(df):
    rows_to_remove = []
    for index, value in df['מחיר'].items():
        text = value.replace('$', '').replace('₪', '').replace(',', '')
        pattern = r'\d{2}/\d{2}/\d{4}'
        match = re.search(pattern, text)
        if match:
            text_price = df.loc[index, 'תאור']
            rooms_match = re.search(r'(\d+(?:\.\d+)?) מחיר',text_price)
            if rooms_match:
                price = float(rooms_match.group(1))
                df.at[index,'מחיר']=price
            else:
                rows_to_remove.append(index)
        if '$' in value:
            df.at[index, 'מחיר'] = int(text) * 3.5
        else:
            try:
                df.at[index, 'מחיר'] = int(text)
                if df.at[index, 'מחיר']<1000 or df.at[index, 'מחיר']>15000:
                    rows_to_remove.append(index)
            except:
                text_price = df.loc[index, 'תאור']
                rooms_match = re.search(r'(\d+(?:\.\d+)?) מחיר',text_price)
                if rooms_match:
                    price = float(rooms_match.group(1))
                    df.at[index,'מחיר']=price
                else:
                    rows_to_remove.append(index)

    df = df.drop(rows_to_remove)
    return df

def chane_rooms(df):
    rows_to_remove = []
    for index, value in df['חדרים'].items():
        pattern = r'\d{2}/\d{2}/\d{4}'
        text = df.loc[index, 'תאור']
        match = re.search(pattern, text)
        if match:
            text = df.loc[index, 'תאור']
            rooms_match = re.search(r'(\d+(?:\.\d+)?) חדרים', text)
            if rooms_match:
                num_rooms = float(rooms_match.group(1))
                df.at[index,'חדרים']=num_rooms
            else:
                rows_to_remove.append(index)
        try:
            stam = int(value)
        except ValueError:
            try:
                stam = float(value)
                if stam>10:
                    rows_to_remove.append(index)
            except ValueError:
                text = df.loc[index, 'תאור']
                rooms_match = re.search(r'(\d+(?:\.\d+)?) חדרים', text)

                if rooms_match:
                    num_rooms = float(rooms_match.group(1))
                    df.at[index,'חדרים']=num_rooms
                else:
                    rows_to_remove.append(index)

    df = df.drop(rows_to_remove)
    return  df


def chane_str(df):
    rows_to_remove = []
    for index, value in df['קומה'].items():
        if value == 'קרקע':
            df.at[index, 'קומה'] = 0
        if value == 'פרטר':
            df.at[index, 'קומה'] = 0.5
        pattern = r'\d{2}/\d{2}/\d{4}'
        text = df.loc[index, 'תאור']
        match = re.search(pattern, text)
        if match:
            text = df.loc[index, 'תאור']
            rooms_match = re.search(r'(\d+(?:\.\d+)?) קומה', text)
            if rooms_match:
                num_rooms = float(rooms_match.group(1))
                df.at[index, 'קומה'] = num_rooms
            else:
                rows_to_remove.append(index)
    for index, value in df['קומה'].items():
        try:
            stam = int(value)
            if stam > 15:
                rows_to_remove.append(index)
        except ValueError:
            try:
                stam = float(value)
                if stam > 15:
                    rows_to_remove.append(index)
            except ValueError:
                text = df.loc[index, 'תאור']
                rooms_match = re.search(r'(\d+(?:\.\d+)?) קומה', text)
                if rooms_match:
                    num_rooms = float(rooms_match.group(1))
                    df.at[index, 'קומה'] = num_rooms
                else:
                    rows_to_remove.append(index)
    df = df.drop(rows_to_remove)
    return df
def bool_to_num(df):
    list_items = ['לשוטפים', 'ריהוט', 'מעלית', 'מזגן', 'חניה', 'מרפסת', 'סורגים', 'ממד', 'מחסן', 'משופצת', 'דוד שמש',
                  'חיות מחמד']
    for item in list_items:
        if item in df.columns:
            df[item] = df[item].astype(int)
        else:
            print(f"Column '{item}' not found in the dataframe.")
    return df
def check_mtrs(df):
    rows_to_remove = []
    for index, value in df['מטרים'].items():
        num_room=float(df.at[index,'חדרים'])
        try:
            stam = int(value)
            if num_room>3 and stam<35:
                rows_to_remove.append(index)
            if stam<25 or stam > 1000:
                rows_to_remove.append(index)
        except ValueError:
            try:
                stam = float(value)
                if num_room>3 and stam<35:
                    rows_to_remove.append(index)
                if stam<25 or stam > 1000:
                    rows_to_remove.append(index)
            except ValueError:
                text = df.loc[index, 'תאור']
                metrs = re.search(r'(\d+(?:\.\d+)?) מטרים|(\d+(?:\.\d+)?) מ"ר', text)
                if metrs: 
                    metr = float(metrs.group(1) or metrs.group(2))
                    if metr<25 or metr > 1000:
                        rows_to_remove.append(index)
                    if num_room>3 and stam<35:
                        rows_to_remove.append(index)
                    df.at[index, 'מטרים'] = metr
                else:
                    rows_to_remove.append(index)
    df = df.drop(rows_to_remove)
    return df
def find_data_desc(df):
    for index, value in df['תאור'].items():
         if str(value) != "nan":
            ans=re.search('חיות מחמד',str(value))
            if ans:
                df.at[index, 'חיות מחמד'] = 1
            else:
                ans=re.search('בעלי חיים',str(value))
                if ans:
                    df.at[index, 'חיות מחמד'] = 1
                else:
                    ans=re.search('בע"ח',str(value))
                    if ans:
                        df.at[index, 'חיות מחמד'] = 1
                    else:
                        ans=re.search('בע"ח',str(value))
                        if ans:
                            df.at[index, 'חיות מחמד'] = 1
    for index, value in df['תאור'].items():
         if str(value) != "nan":
            ans=re.search('לשותפים',str(value))
            if ans:
                df.at[index, 'לשוטפים'] = 1    
    for index, value in df['תאור'].items():
         if str(value) != "nan":
            ans=re.search('לא בעלי חיים',str(value))
            if ans:
                df.at[index, 'חיות מחמד'] =0
            else:
                ans=re.search('ללא חיות מחמד',str(value))
                if ans:
                    df.at[index, 'חיות מחמד'] = 0
                else:
                    ans=re.search('לא בע"ח',str(value))
                    if ans:
                        df.at[index, 'חיות מחמד'] = 0
                    else:
                        ans=re.search('לא חיות',str(value))
                        if ans:
                            df.at[index, 'חיות מחמד'] = 0
                        else:
                            ans=re.search('בלי בעלי',str(value))
                            if ans:
                                df.at[index, 'חיות מחמד'] = 0
                            else:
                                ans=re.search('בלי חיות',str(value))
                                if ans:
                                    df.at[index, 'חיות מחמד'] = 0
                                else:
                                    ans=re.search('לא מתאים לחיות',str(value))
                                    if ans:
                                        df.at[index, 'חיות מחמד'] = 0
                                    else:
                                        ans=re.search('לא מתאים לבעלי חייים',str(value))
                                        if ans:
                                            df.at[index, 'חיות מחמד'] = 0
    return df

my_data=getData('apartaments_ad.csv')
#my_data2=getData('apartaments3.csv')
my_data=my_data.rename(columns={'מכיר':'מחיר'})
#my_data2=my_data2.rename(columns={'מכיר':'מחיר'})
my_data=my_data.dropna()
#my_data2=my_data2.dropna()
my_data=my_data.drop_duplicates()
#my_data2=my_data2.drop_duplicates()
my_data=bool_to_num(my_data)
#my_data2=bool_to_num(my_data2)
my_data=my_data.drop('Unnamed: 0',axis=1)
#my_data2=my_data2.drop('Unnamed: 0',axis=1)
my_data=del_simbols_from_price(my_data)
#my_data2=del_simbols_from_price(my_data2)
my_data=chane_str(my_data)
#my_data2=chane_str(my_data2)
my_data=chane_rooms(my_data)
#my_data2=chane_rooms(my_data2)
my_data=check_mtrs(my_data)
#my_data2=check_mtrs(my_data2)
my_data=find_data_desc(my_data)
my_data.to_csv('data_rdy_to_machine.csv')
#my_data2.to_csv('data_rdy_to_machine1.csv')


# In[49]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error



data = pd.read_csv('data_rdy_to_machine.csv')

data = data.drop(data.columns[0], axis=1)
data= data.drop('תאור', axis=1)
data= data.drop('רחוב', axis=1)


X = data.drop('מחיר', axis=1)  
y = data['מחיר']

X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

model = RandomForestRegressor(400,random_state=420)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R^2 score: {r2}")
print(f"MSE: {mse}")


# In[45]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data_rdy_to_machine.csv')


# Plot the actual price vs predicted price with a linear regression line
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(np.linspace(min(y_test), max(y_test), 100), np.linspace(min(y_test), max(y_test), 100), color='red')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price')
plt.show()

# Histogram of Prices
plt.figure(figsize=(8, 6))
plt.hist(df['מחיר'], bins=20, color='skyblue')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Histogram of Prices')
plt.show()

# Scatter Plot of Square Meters vs. Price
plt.figure(figsize=(8, 6))
plt.scatter(df['מטרים'], df['מחיר'], color='purple')
plt.xlabel('Square Meters')
plt.ylabel('Price')
plt.title('Square Meters vs. Price')
plt.show()

# Box Plot of Price by city
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['עיר'], y=df['מחיר'], palette='Set3')
plt.xlabel('city')
plt.ylabel('Price')
plt.title('Price by city ')
plt.xticks(rotation=90)
plt.show()

# Bar Plot of Average Price by Room Count
avg_price_by_rooms = df.groupby('חדרים')['מחיר'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x=avg_price_by_rooms['חדרים'], y=avg_price_by_rooms['מחיר'], color='green')
plt.xlabel('Room Count')
plt.ylabel('Average Price')
plt.title('Average Price by Room Count')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[50]:



df = pd.read_csv('apartaments_ad.csv')
num_columns = len(df.columns)
num_rows = len(df)

print("Number of columns:", num_columns)
print("Number of rows:", num_rows)

print(df)


# In[32]:


df = pd.read_csv('data_rdy_to_machine.csv')

num_columns = len(df.columns)

num_rows = len(df)

print("Number of columns:", num_columns)
print("Number of rows:", num_rows)
print(df)


# In[ ]:




