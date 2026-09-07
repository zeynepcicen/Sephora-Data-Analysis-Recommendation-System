# Sephora Data Analysis & Recommendation System

> Group Project — May 2026

---

### About the Project

This is an end-to-end data science project developed as a team using a Sephora dataset published on Kaggle. The project consists of 4 main stages: data engineering, exploratory data analysis (EDA), customer segmentation, and a **dual-mode product recommendation system**.

Dataset
The dataset used in this project can be found on Kaggle here.

### Stages

#### 1. Data Engineering
- Raw data containing 10,000+ cosmetic products and over 1 million customer reviews was obtained from Kaggle.
- Missing values were identified and filled — numerical columns with the mean, categorical columns with the mode.
- Outliers were detected using the IQR method and corrected using boundary values.
- Categorical variables such as brand name and skin type were encoded using Label Encoding.
- The cleaned dataset was exported in CSV format to be shared with the team.

#### 2. Exploratory Data Analysis (EDA)
- Brand distributions, product counts by skin type, and the price-rating relationship were visualized.
- The question "Is a more expensive product better?" was analyzed statistically.

#### 3. Customer Segmentation
- Customers were grouped based on purchasing behavior using K-Means.
- Segments such as "budget-conscious buyers" and "luxury brand enthusiasts" were identified.

#### 4. Recommendation System (Dual Mode)

The system operates in two different modes depending on the user's context:

**🔍 Cold Start Mode (Survey Mode)**
Designed for first-time visitors who haven't interacted with any product yet. Based on a few survey questions (average spending, number of different brands tried, skin type, budget):
- The user is first assigned to one of 4 segments: *Budget-Conscious Buyers*, *Luxury Buyers*, *Explorer & Loyal*, *Potential Loyalists*.
- The relevant segment's product pool is filtered by the user's skin type, maximum budget, and minimum rating preference.
- The top 5 highest-rated products are recommended.

**🤖 AI Mode (Content-Based + Segment-Based Ranking)**
Activated when the user has a product they liked or viewed:
- Cosine similarity is used to find the 20 most content-similar products to the selected one.
- This pool is then re-ranked using different criteria depending on the customer's K-Means segment:
  - **Ultra VIP Luxury:** ranked by viral products and value score
  - **Premium Loyalists:** ranked by premium shopping tendency and original price
  - **Balanced Average:** ranked by rating and premium tendency
  - **Other/Fallback:** ranked by overall interaction score
- The final result presents the top 5 best-matching products to the user.

This dual structure allows the system to provide personalized recommendations both for new users (solving the cold-start problem) and for users with prior interaction history.

### Tech Stack
- **Python** — main programming language
- **Pandas / NumPy** — data processing and analysis
- **Scikit-learn** — K-Means clustering, cosine similarity, Label Encoding
- **Matplotlib / Seaborn** — data visualization
- **PyCharm** — development environment

### Team
This project was developed as a group project in May 2026.
