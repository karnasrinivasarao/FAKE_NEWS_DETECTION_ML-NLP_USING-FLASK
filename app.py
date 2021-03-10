from flask import Flask, request, url_for, redirect, render_template,session
import pickle
import pandas as pd
import flask
from flask_cors import CORS
import re
import os
from sklearn.externals import joblib
import numpy as np
import newspaper
from newspaper import Article
import urllib
from flask_mysqldb import MySQL 
import MySQLdb.cursors 

app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')

def clean_article(article):
    art = re.sub("[^A-Za-z0-9' ]", '', str(article))
    art2 = re.sub("[( ' )(' )( ')]", ' ', str(art))
    art3 = re.sub("\s[A-Za-z]\s", ' ', str(art2))
    return art3.lower()


bow = pickle.load(open("bow2.pkl", "rb"))
model2 = pickle.load(open("model1.pkl", "rb"))

#model=pickle.load(open('model.pkl','rb'))
app.secret_key = 'your secret key'
  
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '12345'
app.config['MYSQL_DB'] = 'FND'
  
mysql = MySQL(app)


@app.route('/')
@app.route('/login', methods =['GET', 'POST']) 
def login(): 
    msg = '' 
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form: 
        username = request.form['username'] 
        password = request.form['password'] 
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor) 
        cursor.execute('SELECT * FROM accounts WHERE username = % s AND password = % s', (username, password, )) 
        account = cursor.fetchone() 
        if account: 
            session['loggedin'] = True
            session['id'] = account['id'] 
            session['username'] = account['username'] 
            msg = 'Logged in successfully !'
            return render_template('FND.html') 
        else: 
            msg = 'Incorrect username / password !'
    return render_template('login.html', msg = msg) 
  
@app.route('/logout') 
def logout(): 
    session.pop('loggedin', None) 
    session.pop('id', None) 
    session.pop('username', None) 
    return redirect(url_for('login')) 
  
@app.route('/register', methods =['GET', 'POST']) 
def register(): 
    msg = '' 
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form : 
        username = request.form['username'] 
        password = request.form['password'] 
        email = request.form['email'] 
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor) 
        cursor.execute('SELECT * FROM accounts WHERE username = % s', (username, )) 
        account = cursor.fetchone() 
        if account: 
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email): 
            msg = 'Invalid email address !'
        elif not re.match(r'[A-Za-z0-9]+', username): 
            msg = 'Username must contain only characters and numbers !'
        elif not username or not password or not email: 
            msg = 'Please fill out the form !'
        else: 
            cursor.execute('INSERT INTO accounts VALUES (NULL, % s, % s, % s)', (username, password, email, )) 
            mysql.connection.commit() 
            msg = 'You have successfully registered !'
    elif request.method == 'POST': 
        msg = 'Please fill out the form !'
    return render_template('register.html', msg = msg) 


def main():
    return render_template("FND.html")


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        comment = request.form['article']
        list_comment = [comment]

        list_comment = clean_article(list_comment)
        list_comment = [list_comment]
        vect = bow.transform(list_comment)

        vect = pd.DataFrame(vect.toarray())
        vect.columns = bow.get_feature_names()

        prediction_array = model2.predict(vect)
        proba_array = model2.predict_proba(vect)

        maxProba = np.amax(proba_array)
        maxProba = format(maxProba, ".2%")
        print(maxProba)
        return render_template('FND.html', prediction_text=prediction_array,proba=maxProba)
        
        #url = request.get_data(as_text=True)[5:]
        #url = urllib.parse.unquote(url)
        #article1 = Article(str(url))
        #article1.download()
        #article1.parse()
        #article1.nlp()
        #news = article1.summary
        #prediction=model.predict([news])
        #return render_template('FND.html',prediction_text=prediction)




if __name__ == '__main__':
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)