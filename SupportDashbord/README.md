# Automated Support Data Dashboard

Streamline support workflows with automated data processing and insights.

This repository contains a macro-enabled Excel workbook and Power BI report to automate the ingestion, cleaning, analysis, and visualization of technical support ticket data.

## 📁 Repository Structure

```
SupportDashboard/
├── Excel-Tool/
│   ├── SupportDashboard.xlsm        # Macro-enabled Excel workbook
│   └── Tickets.csv                  # Sample support ticket data
└── PowerBI/
    ├── SupportDashboard.pbix        # Power BI desktop file
    └── README.md                    # instructions for Power BI report
```

## 🛠️ Technologies Used

* **Excel VBA (Microsoft Excel 2016 or later)** for data import, cleaning, pivot refresh, and anomaly alerts. [VBA Reference](https://docs.microsoft.com/office/vba/api/overview/excel)
* **Power BI Desktop (version 2.105.xxxx or later)** for interactive dashboard creation. [Power BI Docs](https://docs.microsoft.com/power-bi/desktop-getting-started)
* **Git (≥2.30) & GitHub** for version control and collaboration. [Git Docs](https://git-scm.com/doc), [GitHub Docs](https://docs.github.com)

## 🚀 Getting Started

### 1. Excel Tool Setup

**Prerequisites:**

* Microsoft Excel 2016 or later (Windows recommended for full VBA support)
* Macros enabled (File → Options → Trust Center → Trust Center Settings → Macro Settings)
* `Tickets.csv` present in the same folder as the workbook

1. Clone or download this repository to your local machine.
2. Open `Excel-Tool/SupportDashboard.xlsm` in Microsoft Excel.
3. Enable macros when prompted.
4. Ensure `Tickets.csv` is located in the same folder as the `.xlsm` file.

### 2. Running the VBA Macros### 2. Running the VBA Macros

* **Import raw data**:

  1. Press `Alt+F8` → select `ImportTickets` → Run.
  2. This will clear and reload `Tickets.csv` into the `RawData` sheet.
* **Clean & transform**:

  1. Press `Alt+F8` → select `CleanTickets` → Run.
  2. This will generate a `CleanData` sheet with proper date formats and a computed `ResolutionDays` column.
* **Refresh pivot tables**:

  1. Press `Alt+F8` → select `RefreshData` → Run.
  2. All pivot tables and charts on the `Analytics` sheet will update.
* **Generate alerts**:

  1. Press `Alt+F8` → select `AnomalyCheck` → Run.
  2. The macro flags tickets with >30 days open and logs them on the `Alerts` sheet.

### 3. Power BI Dashboard

1. Open `PowerBI/SupportDashboard.pbix` in Power BI Desktop.
2. Verify the data source is pointing to the `CleanData` sheet in `SupportDashboard.xlsm`.
3. Refresh the dataset to pull in the latest cleaned data.
4. Explore visuals on the report canvas:

   * **Ticket Volume** by category and month
   * **SLA Compliance** gauge showing on-time vs. overdue tickets
   * **Avg. Resolution Time** by support owner
   * **Open Tickets** anomaly table

### 4. Customization & Testing

* To simulate new data, edit `Tickets.csv` and re-run the `ImportTickets` macro.
* Test error handling by renaming or removing `Tickets.csv` before import.
* Add new pivot chart templates on the `Analytics` sheet and include them in the `RefreshData` macro if needed.

## 🎯 Next Steps

### Short-term Milestones

* Integrate Outlook email automation to send weekly summary reports.
* Add a command-line interface (e.g., PowerShell) to trigger Excel macros externally.

### Long-term Milestones

* Deploy the Power BI report to Power BI Service.
* Configure scheduled refresh.## 🤝 Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

---

*This project was developed as a demonstration of Excel VBA automation and Power BI reporting for technical support workflows.*
