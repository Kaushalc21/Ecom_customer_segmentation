# 📦 E-commerce Customer Segmentation using RFM + Clustering  
### **Data Science | Machine Learning | Customer Analytics**

---

## 📌 **Project Overview**

Customer segmentation is one of the most powerful applications of analytics in e-commerce.  
The goal of this project is to **identify meaningful customer groups** by combining:

- **RFM Analysis** (Recency, Frequency, Monetary)
- **Unsupervised Clustering Models**  
  - K-Means  
  - Agglomerative (Hierarchical) Clustering  
  - DBSCAN  
  - Gaussian Mixture Models (GMM)  

This project mimics a **real industry workflow** used by Amazon, Flipkart, Myntra, and many D2C brands to understand customer behavior and optimize marketing spend.

By segmenting customers based on purchasing patterns, businesses can personalize marketing, improve retention, reduce churn, and increase customer lifetime value (CLV).

---

## 🎯 **Business Motivation**

Most companies lose revenue by treating all customers the same.  
This leads to:

- High marketing cost with low ROI  
- Losing loyal customers due to no targeted engagement  
- No understanding of how customer value changes over time  
- Ineffective discount campaigns  
- Poor personalization  

Customer Segmentation allows companies to:

- Identify **high-value loyal customers**
- Detect **at-risk or inactive customers**
- Target **deal-sensitive buyers**
- Premium-target **big spenders**
- Allocate budget to the **right segment**, not everyone  

This project simulates the exact segmentation strategy used in **real customer analytics teams**.

---

## 🧹 **Data Pre-Processing**

Data preparation is the backbone of reliable segmentation.  
The following steps were performed:

### **1. Data Cleaning**
- Removed duplicate transactions  
- Fixed missing CustomerID entries  
- Handled NULL values in Invoice, Quantity, Description  
- Converted InvoiceDate to datetime  
- Removed negative/cancelled invoices where required

### **2. Feature Engineering**
Created new features including:

- **TotalAmount = Quantity × UnitPrice**  
- Monthly revenue trends  
- Customer-level aggregation  
- Outlier removal for clustering stability  
- RFM feature generation (Recency, Frequency, Monetary)

### **3. Standardization**
Used StandardScaler to normalize RFM values to avoid scale dominance.

---

## 📊 **RFM Analysis**

RFM (Recency-Frequency-Monetary) is a classical marketing technique used to measure customer value.

### **• Recency**  
How recently a customer purchased  
→ Indicates likelihood of engagement

### **• Frequency**  
How often the customer purchases  
→ Indicates brand loyalty & repeat behavior

### **• Monetary**  
How much they spend  
→ Indicates value to the business

RFM gives a **360° view of customer behavior** and becomes the foundation for clustering models.

---

## 🤖 **Clustering Models Used**

### 🔹 **1. K-Means Clustering**
- Most commonly used segmentation model  
- Fast and scalable  
- Works well when clusters are well-separated  

Used on standardized RFM values to derive natural customer groupings.

---

### 🔹 **2. Agglomerative (Hierarchical) Clustering**
Selected as a primary model because:

- It doesn’t assume clusters to be spherical  
- Works extremely well on customer segmentation  
- Provides dendrograms for clear hierarchy  
- Helps understand which customers are similar step-by-step  

This method often performs better than K-Means in noisy datasets.

---

### 🔹 **3. DBSCAN**
Used to:

- Detect noisy customers  
- Identify outliers (e.g., extreme spenders)
- Find customers who behave differently from the main clusters  

Useful when dataset has non-uniform shapes.

---

### 🔹 **4. Gaussian Mixture Model (GMM)**  
**GMM** is an advanced probabilistic clustering algorithm used by companies like Netflix and Amazon for customer behavior modeling.

Why GMM is powerful:

- Allows **soft clustering** – one customer can belong to multiple clusters with probabilities  
- Handles clusters of different shapes  
- Works well with overlapping customer behavior  
- Considered more realistic for marketing segmentation  

**GMM was used to validate whether customers had fuzzy boundaries** (e.g., medium-value customers behaving partly like high-value customers).

---

## 🧠 **Industry Interpretation of Segments**

Typical segments derived:

### **1️⃣ Champions**
- High Monetary  
- High Frequency  
- Recent purchases  
→ These are premium customers; business must retain them.

### **2️⃣ Loyal Customers**
- Buy frequently but may not spend the highest  
→ Offer loyalty programs, exclusive discounts.

### **3️⃣ Potential Loyalists**
- Increasing spend trend  
→ Target with personalized emails.

### **4️⃣ At-Risk Customers**
- No recent purchases  
→ Re-engagement campaigns needed.

### **5️⃣ Low-Value / One-Time Buyers**
- Cheapest customers  
→ Focus minimal marketing resources.

These insights directly support marketing strategy, customer retention, and revenue optimization.

---

## 📈 **Visualizations Created**
Multiple modern visualizations were used:

- RFM Heatmaps  
- RFM Boxplots  
- 3D Clustering Scatterplots  
- Streamlit Dashboard for Interactive Reporting  
- Monthly Revenue Trend Charts  
- Customer Value Density Plots  
- Cluster Distribution Pie Charts  
- Customer Behavior Correlation Maps  

These visuals help decision makers understand patterns at a glance.

---

## 🖥 **Streamlit Dashboard**
A lightweight Streamlit dashboard was built to visualize:

- RFM Score Distribution  
- Cluster-wise segmentation  
- Customer-level drilldowns  
- Monetary distribution  
- Top spending customers  
- Monthly/Weekly trends  

---

## 🖼️ **How to Add Dashboard Screenshots to README**

1. Go to your GitHub Repository  
2. Create a folder (optional)  
