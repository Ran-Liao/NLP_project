#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:51:10 2022

@author: Dogar
"""

import praw
import hashlib
from datetime import datetime, timedelta
import pandas as pd
import os
from praw.models import MoreComments


reddit = praw.Reddit(

client_id="",

client_secret="",

user_agent="test script by modernsufy",

username = "",

password = "" )

print(reddit.user.me())

topics = ["culture", "politics", "religion"]
final_data = pd.DataFrame()

for topic in topics:

    subreddit = reddit.subreddit(topic)
    print(subreddit.display_name)
    # Output: redditdev
    print(subreddit.title)
    # Output: reddit development
    print(subreddit.description)
    # Output: a subreddit for discussion of ...
    
    submissions = []
    for submission in subreddit.top(time_filter="all"):
        print(submission.id)
        submissions.append(submission.id)
    
    dictionary = dict()
    comments = []
    time = []
    for sub in submissions:
        submission = reddit.submission(sub)
        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            comments.append(comment.body)
            time.append(comment.created_utc)
    
    dictionary["Comments"] = comments
    dictionary["datetime"] = time
    
    data = pd.DataFrame(dictionary)
    data['date'] = pd.to_datetime(data['datetime'], unit='s')    
    data = data.sort_values('date')
    data['topic'] = topic
    final_data.append(data)
