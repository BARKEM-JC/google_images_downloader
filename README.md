
# Google Images Downloader

The google image downloaders i found on github didn't work, so i made one for a project, This was migrated from that project in a pretty broken state, tried to rewrite it, broke it even more, but it somewhat works.

*This code does not reflect on my coding ability, it was literally hacked together and didn't take it seriously.*

## What it does

Downloads full scale HQ google images using selenium

Have managed to download up to around 500 images in a single go when it was working properly (trying to return it to that state)

Not super slow but wouldn't call it fast, when multi-processing worked it was fast but oopsie i broke that to.

Is not a software/GUI or CMD, you will need to run it from the project

## What it doesn't do

Hide your ip or use proxies to avoid getting throttled or banned from scraping google images, use at your own risk at its current state

Work well anymore

## Current Features

Download full scale HQ images from google images (Safe search off)

Delete duplicates

## Working on/Planned Features

Go through all the pages/image count without timing out/breaking

Multi-Processing to speed up collection of initial elements

A way to avoid using selenium to lazy load the images

Proxies to avoid bans

Save load system, so you can continue downloading from a point

Multiple queries in one command

Clean up code, i made alot of mistakes and there is alot that can be shortened/changed, i just was mostly testing things out after i migrated it

Require less imports

Fix the text/human detection, maybe replace it with class detection so not limited to only humans

## When to expect a update

Your guess is as good as mine, not really a priority as it served its purpose, if people ask then i might put some more effort in, happy for people to contribute/collaborate

## Use

Personal use to scrape images for projects whether commercial or not, just dont integrate it into a commercial project and im happy

## Project Specific

Keep scraper processes to 1 otherwise you may run into issues, no real speed up yet anyway
Highest ive gotten to work with image_limit is 500 (when it was working *cough* *cough*)

search = Search('eggs', SearchConfig(image_limit=50,  chrome_options=ChromeConfig(DebugMode: bool), scraper_processes=1)

if you find a bunch of chrome/selenium instances stuck after you have ended the script you can uncomment:

kill_all_selenium_instances()
atexit.register(kill_all_selenium_instances)

it will however close all chrome instances including normal browser

Images will be saved in folder GoogleImages