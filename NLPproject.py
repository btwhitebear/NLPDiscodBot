import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
from sklearn.metrics import precision_score, recall_score
import discord
from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

with open("hate_speech_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

data = pd.read_csv('labeled_data.csv')

text_data = data['tweet']

labels = data['class']

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

text_data = text_data.apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train_vec, y_train)

predicted_y = model.predict(X_test_vec)


accuracy = svm_classifier.score(X_test_vec, y_test)
precision = precision_score(y_test, predicted_y, average='weighted')
recall = recall_score(y_test, predicted_y, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

with open('hate_speech_model.pkl', 'wb') as model_file:
    pickle.dump(svm_classifier, model_file)

user_warnings = {}

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return 
    processed_message = preprocess_text(message.content)
    message_vec = vectorizer.transform([processed_message])
    prediction = model.predict(message_vec)
    if prediction == 0:  
        user_id = message.author.id
        if user_id in user_warnings:
            user_warnings[user_id] += 1
        else:
            user_warnings[user_id] = 1
        if user_warnings[user_id] == 2:
            await message.author.send("You have been banned for repeated hate speech violations.")
            # Uncomment the line below to actually ban the user
            # await message.author.ban(reason="Repeated hate speech violations")
        elif user_warnings[user_id] < 2:
            await message.author.send("Warning: Your message may contain hate speech.")
    else:
        await bot.process_commands(message)

@bot.command() 
async def unbanall(ctx):
    bans = await ctx.guild.bans()
    for ban in bans:
        await ctx.guild.unban(ban.user)
    await ctx.send("Unbanned all users.")

@bot.command()
async def reset(ctx):
    user_warnings.clear()
    await ctx.send("Warnings have been reset.")

bot.run('') #discord bot token goes here