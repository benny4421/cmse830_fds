# **Car Crash Data Analysis**

An interactive Streamlit dashboard for exploring demographic disparities in U.S. crash injury outcomes using NEMSIS data (2018-2022).

## **Live Application**

The interactive dashboard is publicly deployed and can be accessed directly via the following link:

[**https://cmse830fds-3zuewsuxruovjwtxwiyqh9.streamlit.app/**](https://cmse830fds-3zuewsuxruovjwtxwiyqh9.streamlit.app/)

## **Application Structure (Table of Contents)**

The dashboard is organized into a logical narrative that walks you through the data analysis process. You can navigate through the pages using the sidebar on the left.

1. **Overview**: Introduces the project's goals, target audience (policymakers, safety agencies), and provides a preview of the dataset.  
2. **Handling Data Duplicates**: Details the investigation into semantic duplicate records and the data cleaning strategy.  
3. **Handling Missing Values**: Explains the process of identifying and standardizing both standard (NaN) and semantic ("unknown", "not recorded") missing values.  
4. **US Census Data Merging**: Describes the next step in the analysis—normalizing the crash data with population counts for a more accurate assessment.  
5. **Visualization**: Presents the key preliminary findings through a series of interactive charts.

## **Interactive Features**

This dashboard is built for exploration. Here are some of the key interactive features you can use:

* **Collapsible Navigation Bar**: The page navigation menu on the left can be collapsed by clicking the "\>" icon at the top, providing a wider view of the content.  
* **Interactive Plots**: All charts on the **Visualization** page (and most others, excluding the static missing value heatmaps) are fully interactive.  
  * **Toggle Elements**: Click on items in the legend of any plot to hide or show specific data series. This is useful for comparing specific groups.  
  * **Zoom and Pan**: Click and drag your mouse over a section of a chart to zoom in for a more detailed view. Double-click to reset the view.  
  * **Detailed Hover Information**: Hover your mouse over any data point on a chart to see its precise value and details.
