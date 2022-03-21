#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/robgon-art/MAGnet/blob/main/1_MAGnet_Gather_Modern_Paintings.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>x


# In[ ]:
import os

cwd = os.getcwd() #returns the current directory that the file is working in  a
#string format for the running the script and save the images in the same images
#into my own directory

file_path = cwd+"/modern_paintings"


# In[ ]:
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
shell = InteractiveShell.instance()
get_ipython().system('mkdir $file_path')

# In[ ]:

import urllib #collects several modules for working with URLs
import re #regular expression matching operations similar
from bs4 import BeautifulSoup #pulling data out of HTML and XML files, works with parser
#to provide idiomatic ways of navigating, searching, and modifying the parse tree
import time #provides various time-related functions

def get_images(url):
  print(url)
  genre_soup = BeautifulSoup(urllib.request.urlopen(url, "lxml")) # 1. represents
  #the document as a nested data structure; "lxml" - parser
  #2. defines functions and classes which help in opening URLs (mostly HTTP)
  #in a complex world — basic and digest authentication, redirections, cookies and more
  artist_list_main = genre_soup.find("main")
  lis = artist_list_main.find_all("li")
  # The script goes through each artist on the site alphabetically. It checks to
  # see if the artist is tagged as part of the “modern” art movement and if they
  # were born after 1800 and died before 1950.

  # for each list element
  for li in lis:
    born = 0
    died = 0

    # get the date range
    for line in li.text.splitlines():
      if line.startswith(",") and "-" in line:
        parts = line.split('-')
        if len(parts) == 2:
          born = int(re.sub("[^0-9]", "",parts[0]))
          died = int(re.sub("[^0-9]", "",parts[1]))

    # look for artists who may have created work that could in public domain
    if born>1800 and died>0 and died<1950:
      link = li.find("a")
      artist = link.attrs["href"]

      # get the artist's main page
      artist_url = base_url + artist
      artist_soup = BeautifulSoup(urllib.request.urlopen(artist_url), "lxml")

      # only look for artists with the word modern on their main page
      if "modern" in artist_soup.text.lower():
        print(artist + " " + str(born) + " - " + str(died))

        # get the artist's web page for the artwork
        url = base_url + artist + '/all-works/text-list'
        artist_work_soup = BeautifulSoup(urllib.request.urlopen(url), "lxml")

        # get the main section
        artist_main = artist_work_soup.find("main")
        image_count = 0
        artist_name = artist.split("/")[2]

        # get the list of artwork
        lis = artist_main.find_all("li")

        # for each list element
        for li in lis:
          link = li.find("a")

          if link != None:
            painting = link.attrs["href"]

            # get the painting
            url = base_url + painting
            print(url)

            try:
              painting_soup = BeautifulSoup(urllib.request.urlopen(url), "lxml")

            except:
              print("error retreiving page")
              continue

            # check the copyright
            if "Public domain" in painting_soup.text:

              # get the url
              og_image = painting_soup.find("meta", {"property":"og:image"})
              image_url = og_image["content"].split("!")[0] # ignore the !Large.jpg at the end
              print(image_url)

              parts = url.split("/")
              painting_name = parts[-1]
              save_path = file_path + "/" + artist_name + "_" + painting_name + ".jpg"

              #download the file
              try:
                print("downloading to " + save_path)
                time.sleep(0.2)  # try not to get a 403
                urllib.request.urlretrieve(image_url, save_path)
                image_count = image_count + 1
              except Exception as e:
                print("failed downloading " + image_url, e)


# In[ ]:


base_url = "https://www.wikiart.org"
urls = []
GG_terms = ['abstract-art', 'surrealism', 'abstract-expressionism',
             'post-painterly-abstraction', 'pop-art',
            'minimalism','post-minimalism','photorealism',
           'contemproray','conceptual-art','contemporary-realism',
            'hyper-mannerism-anachronism','documentary-photography',
          'street-photography']

# 'color-field-painting', 'hard-edge-painting',
#for c in range(ord('a'), ord('z') + 1)#
for c in GG_terms:
  #char = chr(c)
  artist_list_url = base_url + "/en/artists-by-art-movement/" + c + "#!#resultType:text"
  urls.append(artist_list_url)

print(urls)

# In[ ]:


from concurrent.futures import ThreadPoolExecutor
executor = None
with ThreadPoolExecutor(max_workers = 8) as executor:
  ex = executor
  executor.map(get_images, urls)
