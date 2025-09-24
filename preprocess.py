import json
import sys
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from llm_helper import llm
# Force stdout to use UTF-8 so ✔ and other symbols won't break
sys.stdout.reconfigure(encoding="utf-8")

def process_posts(raw_file_path, processed_file_path="data/processed_posts.json"):
    enriched_posts = []
    with open(raw_file_path, encoding="utf-8") as file:
        posts = json.load(file)
        for post in posts:
            metadata = extract_metadata(post['text'])
            post_with_metadata = post | metadata
            enriched_posts.append(post_with_metadata)
 
    # ✅ Correct: print enriched posts, not the old ones
    for epost in enriched_posts:
        print(json.dumps(epost, ensure_ascii=False, indent=2))

    # ✅ Save enriched posts, not the old ones
    with open(processed_file_path, "w", encoding="utf-8") as outfile:
        json.dump(enriched_posts, outfile, ensure_ascii=False, indent=2)

        unified_tags=get_unified_tags(enriched_posts)
    
    for post in enriched_posts:
        current_tags=post['tags']
        new_tags={unified_tags[tag] for tag in current_tags}
        post['tags']=list(new_tags)
    with open(processed_file_path,encoding='utf-8',mode="w") as outfile:
        json.dump(enriched_posts , outfile, indent=4)

def get_unified_tags(posts_with_metadata):
    unique_tags=set()
    for post in posts_with_metadata:
        unique_tags.update(post['tags'])
    
    unique_tags_list=json.dumps(list(unique_tags))

    template='''I will give you a list of tags. you need to unify tags wit the following requirements,
    1.Tags are unified are merged to create a shorter list.
      Example 1:"Jobseekers", "Job Hunting" can be all merged into a single tag "Job search".
      Example 2:"Motivation",Inpiration","Drive" can be mapped to "Motivation"
      Example 3:"Personal Growth","Personal Developement", "Self Improvement" can be mapped to "SelfImprovement"
      Example 4:"Scam Alert","Job Scam" etc. can be mapped to "Scams"
    2.Each tag should be follow title case convention. example:"Motivation","Job Search"
    3.Output should be a JSON object, No preamble
    4.Output should have mapping of original tag and the unified tag.
      For example:{{"Jobseekers":"Job search","Job Hunting":,"Job Search","Motivation":"Motivtion"}} 
       
    Here is the list of tags:
    {tags}
    '''

    pt=PromptTemplate.from_template(template)
    chain=pt|llm
    response=chain.invoke(input={"tags":str(unique_tags_list)})
    try:
        json_parser=JsonOutputParser()
        res=json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context is too big. Unable to parse jobs.")
    return res

def extract_metadata(post):
    template='''Your are given a linkedin post. you need to extract number of lines, language of the post and tags.
    1. return a valid JSON. No preamble.
    2. JSON object should have exactly three keys: line_count,language and tags.
    3. tags is an array of text tags. Extract maximum two tags.
    4. Language should be English or Hinglish (Hinglish means hindi + english)
     
     
     here is the actual post on which you need to perform this task:
    {posts}
    '''
    pt=PromptTemplate.from_template(template)
    chain=pt | llm
    response=chain.invoke(input={'posts':post})

    try:
        json_parser=JsonOutputParser()
        res=json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context too big,Unable to parse jobs")
    return res


if __name__ == "__main__":
    process_posts("data/raw_posts.json", "data/processed_posts.json")
