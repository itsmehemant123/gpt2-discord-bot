import re
import io
import os
import sys
import json
import time
import discord
from datetime import datetime, timedelta
from discord.ext import commands
from discord import utils
import logging
from src import model, sample, encoder
import numpy as np
import tensorflow as tf

class GPT2Bot(commands.Cog):

    def __init__(self, bot):
        logging.basicConfig(level=logging.INFO)
        
        self.bot = bot
        self.reset_model()
    
    def init_state(self):
        self.model_name='117M'
        self.batch_size = 1
        self.seed = 42069
        self.nsamples=1
        self.length=10
        self.temperature=1
        self.top_k=0
    
    def set_state(self, nsamples, length, temperature, top_k):
        self.nsamples = nsamples
        self.length = length
        self.temperature = temperature
        self.top_k
    
    def preinit_model(self):
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.enc = encoder.get_encoder(self.model_name)
        self.hparams = model.default_hparams()
        with open(os.path.join('models', self.model_name, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))

        if self.length is None:
            self.length = self.hparams.n_ctx // 2
        elif self.length > self.hparams.n_ctx:
            logging.error("Can't get samples longer than window size: %s" % self.hparams.n_ctx)
    
    def init_model(self):
        self.context = tf.placeholder(tf.int32, [1, None])
        self.output = sample.sample_sequence(
            hparams=self.hparams, length=self.length,
            context=self.context,
            batch_size=self.batch_size,
            temperature=self.temperature, top_k=self.top_k
        )

        self.saver = tf.train.Saver()
        self.ckpt = tf.train.latest_checkpoint(os.path.join('models', self.model_name))
        self.saver.restore(self.session, self.ckpt)

    def reset_model(self):
        self.init_state()
        self.preinit_model()
        self.session = tf.InteractiveSession(graph=tf.Graph())
        self.init_model()

    def shutdown(self):
        logging.info('Shutting down GPT.')
        self.session.close()

    @commands.command()
    @commands.guild_only()
    async def talk(self, ctx, *, message):
        logging.info('MSG: ' + message)
        context_tokens = self.enc.encode(message)
        for _ in range(self.nsamples):
            out = self.session.run(self.output, feed_dict={
                self.context: [context_tokens for _ in range(1)]
            })[:, len(context_tokens):]
            resp_text = self.enc.decode(out[0])
            logging.info('RESPONSE:' + resp_text)
            await ctx.send(resp_text)

    @commands.command()
    @commands.guild_only()
    async def getconfiguration(self, ctx):
        logging.info('Current state.')
        await ctx.send('N Samples: ' + str(self.nsamples))
        await ctx.send('Max Length: ' + str(self.length))
        await ctx.send('Temperature: ' + str(self.temperature))
        await ctx.send('Top K: ' + str(self.top_k))

    @commands.command()
    @commands.guild_only()
    async def helpconfigure(self, ctx):
        logging.info('Help Invoked.')
        await ctx.send('Configure the bot session by `!configure <nsamples> <length> <temperature> <topk>`.')
        await ctx.send('Get current state by `!getconfiguration`.')

    @commands.command()
    @commands.guild_only()
    async def configure(self, ctx, nsamples, length, temp, top_k):
        logging.info('Set configuration.')
        await ctx.trigger_typing()
        self.shutdown()
        self.set_state(int(nsamples), int(length), int(temp), int(top_k))
        await ctx.trigger_typing()
        self.preinit_model()
        self.session = tf.InteractiveSession(graph=tf.Graph())
        await ctx.trigger_typing()
        self.init_model()

        await ctx.send('Succesfully Set Configuration!')

    @commands.command()
    @commands.guild_only()
    async def default(self, ctx):
        logging.info('Set configuration.')
        await ctx.trigger_typing()
        self.shutdown()
        self.init_state()
        await ctx.trigger_typing()
        self.preinit_model()
        self.session = tf.InteractiveSession(graph=tf.Graph())
        await ctx.trigger_typing()
        self.init_model()

        await ctx.send('Succesfully Set Configuration!')
    
        
def setup(bot):
    bot.add_cog(GPT2Bot(bot))
