import json
import os
import random
from json import JSONDecodeError

from slugify import slugify
from datetime import datetime

INPUT_FILE = "dblp_v14.json"
N = 5

papers_node = []
writes_edge = set()
cites_edge = set()
about_edge = set()
publication_edge = []

# New ids for nodes
authors_map = {}  # authors id, name
papers_map = {}  # paper id, title
topics_map = {}  # topic id, name
journals_map = {}  # journals id, title

authors_id = {}
next_author_id = -1
authors_name_to_id = {}

# topics_map is also topics_id, because topic is identified by its name
# next_topic_id = -1
# topics_count = {}

# journals_map is also journals_id, because journal is identified by its name
next_journal_id = -1


def get_author_id(author):
    verbose_id = author['id'].strip()
    author_name = author['name'].replace('"', "'").title()

    if verbose_id == "":
        verbose_id = slugify(author['name'])
    if author_name == "":
        author_name = "N/A"

    id = authors_id.get(verbose_id)
    if id:
        if not authors_name_to_id.get(author_name):
            authors_name_to_id[author_name] = id
        if authors_map[id] == "N/A":
            authors_map[id] = author_name
        return id
    else:
        id = authors_name_to_id.get(author_name)
        if id and author_name != "N/A":
            return id
        else:
            global next_author_id
            next_author_id += 1
            authors_id[verbose_id] = next_author_id
            authors_name_to_id[author_name] = next_author_id
            authors_map[next_author_id] = author_name
            return next_author_id


# def get_topic_id(topic):
#     topic = topic.strip().title()
#     id = topics_map.get(topic)
#     if id:
#         topics_count[id] += 1
#         return id
#     else:
#         global next_topic_id
#         next_topic_id += 1
#         topics_count[next_topic_id] = 0
#         topics_map[topic] = next_topic_id
#         return next_topic_id


def get_journal_id(journal):
    journal = journal.strip().title()
    id = journals_map.get(journal)
    if id:
        return id
    else:
        global next_journal_id
        next_journal_id += 1
        journals_map[journal] = next_journal_id
        return next_journal_id


# def get_paper_id(verbose_paper_id, paper_name):
#     verbose_id = verbose_paper_id.strip()
#     id = papers_id.get(verbose_id)
#     if id:
#         if paper_name is not None:
#             papers_map.append(f"{id},\"{paper_name}\"")
#         return id
#     else:
#         global next_paper_id
#         next_paper_id += 1
#         papers_id[verbose_id] = next_paper_id
#         if paper_name is not None:
#             papers_map.append(f"{next_paper_id},\"{paper_name}\"")
#         return next_paper_id


def extract_dataset(root):
    time_1 = datetime.now()
    with open(INPUT_FILE, "r") as input_file:
        for input_line in input_file:
            try:
                input_paper = json.loads(input_line[:-2])
            except JSONDecodeError:
                continue

            # Papers
            # paper_id = get_paper_id(input_paper['id'], input_paper['title'])
            paper_id = input_paper['id']
            papers_map[(paper_id, f"{input_paper['title']}")] = input_paper['year']
            papers_node.append((paper_id, f"{input_paper['n_citation']},{input_paper['year']}"))

            if input_paper.get('doc_type').lower() == "journal":
                # Publication
                input_journal = input_paper['venue'].get('raw')
                if input_journal:
                    journal_id = get_journal_id(input_journal)
                    publication_edge.append((paper_id, journal_id))

            if input_paper.get('fos'):
                # About
                for input_topic in input_paper['fos']:
                    # topic_id = get_topic_id(input_topic['name'])
                    if topics_map.get(input_topic['name'].strip().title()):
                        topics_map[input_topic['name'].strip().title()] += 1
                    else:
                        topics_map[input_topic['name'].strip().title()] = 1
                    about_edge.add((paper_id, input_topic['name'].strip().title()))

            if input_paper.get('keywords'):
                # About
                for input_topic in input_paper['keywords']:
                    # topic_id = get_topic_id(input_topic)
                    if topics_map.get(input_topic.strip().title()):
                        topics_map[input_topic.strip().title()] += 1
                    else:
                        topics_map[input_topic.strip().title()] = 1
                    about_edge.add((paper_id, input_topic.strip().title()))

            # Cites
            if input_paper.get('references'):
                for input_citation in input_paper['references']:
                    # citation_paper_id = get_paper_id(input_citation, None)
                    cites_edge.add((paper_id, input_citation))

            # Authors & Writes
            for input_author in input_paper['authors']:
                author_id = get_author_id(input_author)
                writes_edge.add((author_id, paper_id))

    time_2 = datetime.now()
    print(f"Parsing time: {str(time_2 - time_1)}")

    # root = time_2.strftime("%Y%m%d%H%M%S")
    if not os.path.exists(f"{root}"):
        os.mkdir(f"{root}")
    if not os.path.exists(f"{root}/mapping"):
        os.mkdir(f"{root}/mapping")
    if not os.path.exists(f"{root}/raw"):
        os.mkdir(f"{root}/raw")
    if not os.path.exists(f"{root}/split/"):
        os.mkdir(f"{root}/split/")

    # Mapping
    authors_file = open(f"{root}/mapping/authors.csv", "w", encoding="utf-8")
    authors_file.write("id,name\n")
    for id, name in authors_map.items():
        authors_file.write(f"{id},\"{name}\"\n")
    authors_file.close()

    papers_file = open(f"{root}/mapping/papers.csv", "w", encoding="utf-8")
    papers_file.write("id,title\n")
    papers_id = {}
    sorted_papers_list = sorted(papers_map.items(), key=lambda x: x[1])
    for i, ((id, title), _) in enumerate(sorted_papers_list):
        papers_id[id] = i
        papers_file.write(f"{i},\"{title}\"\n")
    papers_file.close()

    topics_file = open(f"{root}/mapping/topics.csv", "w", encoding="utf-8")
    topics_file.write("id,name\n")
    new_topic_ids = {}
    new_topic_id = 0
    for name, count in topics_map.items():
        if count > 10:
            new_topic_ids[name] = new_topic_id
            topics_file.write(f"{new_topic_id},\"{name}\"\n")
            new_topic_id += 1
    topics_file.close()

    journals_file = open(f"{root}/mapping/journals.csv", "w", encoding="utf-8")
    journals_file.write("id,name\n")
    for name, id in journals_map.items():
        journals_file.write(f"{id},\"{name}\"\n")
    journals_file.close()

    # Raw
    papers_file = open(f"{root}/raw/papers.csv", "w", encoding="utf-8")
    papers_file.write("id,citations,year\n")
    for id, info in papers_node:
        papers_file.write(f"{papers_id[id]},{info}\n")
    papers_file.close()

    writes_file = open(f"{root}/raw/writes.csv", "w", encoding="utf-8")
    writes_file.write("author_id,paper_id\n")
    for author_id, paper_id in writes_edge:
        writes_file.write(f"{author_id},{papers_id[paper_id]}\n")
    writes_file.close()

    cites_file = open(f"{root}/raw/cites.csv", "w", encoding="utf-8")
    cites_file.write("paper1_id,paper2_id\n")
    for paper1_id, paper2_id in cites_edge:
        try:
            cites_file.write(f"{papers_id[paper1_id]},{papers_id[paper2_id]}\n")
        except:
            continue
    cites_file.close()

    about_file = open(f"{root}/raw/about.csv", "w", encoding="utf-8")
    about_file.write("paper_id,topic_id\n")
    for paper_id, topic_id in about_edge:
        try:
            about_file.write(f"{papers_id[paper_id]},{new_topic_ids[topic_id]}\n")
        except Exception:
            pass
    about_file.close()

    publication_file = open(f"{root}/raw/publication.csv", "w", encoding="utf-8")
    publication_file.write("paper_id,journal_id\n")
    for paper_id, journal_id in publication_edge:
        publication_file.write(f"{papers_id[paper_id]}, {journal_id}\n")
    publication_file.close()

    # split into n authors fold
    author_list = list(authors_id.values())
    random.Random(561).shuffle(author_list)

    author_split_file = []
    for i in range(N):
        author_split_file.append(open(f"{root}/split/authors_{i}.csv", "w"))

    for i, author in enumerate(author_list):
        n_spilt = i % N
        author_split_file[n_spilt].write(f"{author}\n")

    for file in author_split_file:
        file.close()

    print(f"Total time: {str(datetime.now() - time_1)}")
