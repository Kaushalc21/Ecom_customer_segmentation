# E-commerce Customer Segmentation using RFM, Clustering & Behavioral Analytics

---

## 📌 Project Overview
This project focuses on **segmenting e-commerce customers** based on their purchasing behavior using  
**RFM Analysis (Recency, Frequency, Monetary)** combined with **unsupervised clustering** techniques such as:

- Agglomerative Clustering  
- K-Means  
- Gaussian Mixture Models (GMM)  

The segmentation output helps businesses understand **customer value, loyalty, purchase intent, and revenue contribution**.  
The insights can be directly used for marketing personalization, retention strategies, and targeted promotions.

![Screenshot](imgs/Screenshot%202025-12-08%20114919.png)

---

## 🧩 Business Objective
Modern e-commerce platforms generate large amounts of transactional data.  
However, treating all customers the same often results in:

- Wasted marketing budget  
- Poor engagement  
- Low retention  
- Missed high-value opportunities  

**Customer segmentation solves this by grouping customers based on real purchasing behavior**, enabling:

- Identification of high-value “Champion” customers  
- Re-engagement strategies for inactive customers  
- Personalized discounting for sensitive buyers  
- Improved ROI from marketing campaigns  
- Better inventory & product planning  

This segmentation gives the business **a data-driven understanding of their customer lifecycle** rather than guessing.

---

## 🛠️ Data Pipeline & Preprocessing

### **1. Data Cleaning**
- Removed duplicate transactions  
- Imputed or removed missing values  
- Converted InvoiceDate to datetime  
- Calculated *TotalPrice = Quantity × UnitPrice*  
- Filtered canceled/negative invoices  

### **2. Feature Engineering**
- Constructed **RFM matrix**  
- Standardized features using MinMax/StandardScaler  
- Created customer-level aggregated dataset  

### **3. Exploratory Data Analysis (EDA)**
Analyzed:

- Monthly revenue trends  
- Top selling products  
- Customer purchasing patterns  
- Outliers in spending behavior  
- Seasonal effects on sales  

This step ensures we understand **how customers behave before clustering them**.

---

## 📊 RFM Analysis (Recency, Frequency, Monetary)

RFM is a proven industry standard used by Amazon, Walmart, and most e-commerce analytics teams.

- **Recency (R):** How recently a customer purchased  
- **Frequency (F):** How often they purchased  
- **Monetary (M):** How much they spent  

Each value reveals a different part of the customer lifecycle.


<p align="center">
  <img src="imgs/Screenshot 2025-12-08 115735" width="30%">
</p>

RFM helps businesses:

- Identify VIP customers  
- Detect churn risks  
- Create loyalty programs  
- Understand high vs low spenders  
- Prioritize marketing budget  

---

## 🤖 Clustering Techniques Used

### ### 1️⃣ **K-Means Clustering**
- Fast, scalable  
- Works well with normalized continuous features  
- Creates spherical clusters  
- Good for RFM datasets  

### ### 2️⃣ **Agglomerative Clustering**
- Hierarchical clustering  
- Does not assume cluster shape  
- Useful for visual dendrogram-based segmentation  
- More interpretable than KMeans  

### ### 3️⃣ **Gaussian Mixture Model (GMM)**
GMM assumes **each cluster is a probability distribution**, allowing:

- Soft clustering (customer belongs to segments with probabilities)  
- More flexible cluster shapes  
- Better modeling for real-world non-linear spending behavior  

GMM is widely used in industry because **customer behavior is rarely perfectly separated**.

Example:
A customer might be:
- 70% likely a “Loyal Buyer”
- 30% likely a “Potential Loyalist”

Soft clustering provides richer understanding for personalized targeting.


<p align="center">
  <img src="imgs/Screenshot 2025-12-08 115652" width="30%">
</p>

---

## 🧠 Why Segmentation Matters (Explained)


<p align="center">
  <img src="imgs/Screenshot 2025-12-08 115735..png" width="70%">
</p>

> “This project converts raw transactional data into business-driven customer intelligence.  
> Using RFM and clustering models (KMeans, Hierarchical, and GMM), I segmented customers into behavioral groups such as Champions, Inactive Users, One-Time Buyers, and Loyal Customers.  
> These insights help reduce marketing costs, improve customer retention, and identify revenue-driving groups.  
> The project demonstrates the complete journey from data cleaning → feature engineering → segmentation → dashboard visualization, similar to real-world e-commerce analytics pipelines.”


