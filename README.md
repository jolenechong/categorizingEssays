# Multi-Class Text Classification: Categorizing Essays

Problem: 📚 Education Industry, Tuition Centres have a database of past well-written student essays that could be used as reference for future students.

Our objective is to create an AI model to automate the categorization of essays at 75% accuracy for easier retrieval and reference, and ultimately for students to gain access to a wealth of valuable essays.

The AI model utilizes both LDA topic modelling and multi-class text classification techniques.

Date: February 2023<br/>
Live site: https://categorizing-essays-ai.streamlit.app/ <br/><br/>

<img src="assets/Demo.png" alt="Screenshot of the results of the AI on Streamlit"/>


## How it works

#### Data Scraping
I first scraped 2.5k rows of data per category from GradesFixer's Essay Examples Website. As essays are long, much more time is needded to preprocess the text hence we've decided on a goal of 14k cleaned essays. GradesFixer's Essay Examples

These were the decided categories after looking at websites with essay samples, ensuring distinct categories. I also removed duplicates and essays with multiple overlapping categories before ensuring even distribution among categories.

<img src="assets/distributionOfCategories.png" alt="Pie Chart of the distribution of categories"/> <br/>

#### Data Cleaning and Preprocessing
To clean the data, we would need to first understand what is needed. So I started with visualizing data to understand columns, null, duplicates and more.

Steps to Data Cleaning
- Handled Nulls (removed), Duplicates (none)
- Language Detection (remove noise of non-english data)
- Did the following on all text columns
- Whitespace Tokenization
- Negation Handling
- Expanded Contractions
- Removing diacritics (there were many in history essays, helps to decrease dimensionality)
- POS Tagging
- Normalized Text (case conversion, strip punctuation, special characters, numbers and stopwords
- Lemmatization with POS Tagging (to optimize)
- Generated Noun-Only and Summary-Only (LSA vs TextRank) datasets to identify main ideas and themes of the essay

Here are 2 of the most interesting insights gained.

Literature, Social Issues and Life Essays tend to have the lowest word count
<img src="assets/categoriesWithLongestEssays.png" alt="Pie Chart of the distribution of categories"/> <br/>

"People" was one of the highly important repeated words across categories. Other than that, the categories and their top words correspond with each other.
<img src="assets/wordCloud.png" alt="Word Cloud categorized by categories"/> <br/>

Let's now move on to the modelling process.

## The Process


## Contact
Jolene - [jolenechong7@gmail.com](mailto:jolenechong7@gmail.com) <br>