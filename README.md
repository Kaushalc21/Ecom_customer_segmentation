# 🛒 E-Commerce Customer Segmentation using RFM, Clustering & GMM

## 📌 Project Overview
This project performs **data-driven customer segmentation** using a combination of:
- **RFM Analysis** (Recency, Frequency, Monetary)
- **Agglomerative Clustering**
- **K-Means Clustering**
- **Gaussian Mixture Models (GMM)**

The objective is to help an e-commerce company better understand customer purchasing behavior and identify **actionable customer segments** for marketing, retention, and revenue optimization.

---

## 🚀 Business Problem
Most businesses treat all customers the same. This leads to:
- High marketing costs  
- Poor customer retention  
- Low ROI on promotions  
- Missed opportunities to re-engage high-value customers  

Customer segmentation solves these problems by enabling:
- Identification of **champion customers**
- Recognition of **at-risk or inactive users**
- Personalized marketing strategies
- Data-backed decision making

---

## 🧹 Data Preparation & Feature Engineering

### ✔️ Data Cleaning  
- Removed duplicates  
- Handled missing values (CustomerID, InvoiceDate issues)  
- Converted datatypes (dates → datetime)  
- Removed cancellations/negative quantities  

### ✔️ Feature Engineering  
Created the following features:
- **Recency** — Days since last purchase  
- **Frequency** — Number of unique invoices  
- **Monetary** — Total spending  

RFM metrics were standardized using **MinMaxScaler** for clustering compatibility.

---

## 📊 Exploratory Data Analysis (EDA)
EDA focused on understanding:
- Distribution of transactions over time  
- Revenue trends  
- Customer purchase patterns  
- Outliers in monetary spending  
- Monthly revenue and purchase cycles  

Visualizations included:
- Line charts  
- Histograms  
- Boxplots  
- Treemaps  
- 3D RFM visualization  

---

## 🤖 Machine Learning Models Used

### **1️⃣ Agglomerative Clustering**
Hierarchical clustering used due to:
- High interpretability  
- Dendrogram visualization  
- Ability to capture non-spherical clusters  

Generated segments:
- **Champions**
- **Potential Loyalists**
- **Regular Customers**
- **Low-Value / One-Time Buyers**

---

### **2️⃣ K-Means Clustering**
Used to validate the cluster structure learned from RFM.  
Advantages:
- Fast and scalable  
- Performs well on standardized numerical data  

Elbow method + Silhouette Score used for optimal k selection.

---

### **3️⃣ Gaussian Mixture Model (GMM)**  
GMM assumes **soft clustering**, meaning a customer can belong to more than one segment with probability.

Why GMM was included:
- Handles overlapping clusters better  
- Works well with non-linear distributions  
- Provides probability distribution for each segment  
- More realistic modeling of customer behavior  

This adds **depth and ML maturity** to your project — a very strong point for interviews.

---

## 🧠 Why RFM + Clustering?
RFM gives **customer-level business metrics**, while clustering groups customers based on similarity.

This hybrid approach provides:
- Behavioral segmentation  
- Monetary-based insights  
- Identification of valuable customer cohorts  
- Actionable marketing insights  

Used widely in:
- E-commerce  
- Retail analytics  
- Subscription platforms  
- CRM systems  

---

## 📈 Dashboard (Streamlit)
A Streamlit dashboard was created to make the project interactive:
- RFM sliders for dynamic filtering  
- Cluster comparison views  
- Customer-level insights  
- Revenue trend visualization  

### 📸 Add Dashboard Screenshots  
Create a folder called `images/` and place your dashboard pictures:

