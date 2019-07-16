import discord
import json
import logging
import inspect
from discord.ext import commands
from gptchatbot import GPT2Bot

logging.basicConfig(level=logging.INFO)

with open('./config/auth.json') as data_file:
    auth = json.load(data_file)

bot = commands.Bot(command_prefix=commands.when_mentioned_or('!'), description='GPT-2', max_messages=5000)
# gptCog = GPT2Bot(bot)
# bot.add_cog(gptCog)
initial_extensions = ['gptchatbot']

if __name__ == '__main__':
    for extension in initial_extensions:
        try:
            bot.load_extension(extension)
        except Exception as e:
            logging.error(f'Failed to load extension {extension}. Error: {e}')


@bot.event
async def on_ready():
    await bot.change_presence(activity=discord.Game(name='MayQueen Nyan²', type=0, url='https://github.com/KaitoCross/gpt2-discord-bot'))
    logging.info('Logged in as:{0} (ID: {0.id})'.format(bot.user))

bot.run(auth['token'])