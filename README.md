# Optimal Matching (OM) Sequence Analysis Use Cases | 2021 NCCU MIS Thesis
- Thesis Name: `Optimal Matching & Sequence Clustering: Analyses of E-Commerce User Behaviors & Delivery Logistics`
- 論文名稱: `最佳匹配法與序列分群: 電商用戶行為與運輸物流的分析`

### Thesis Advisors and Oral Examination Committee
- [Dr. Hao-Chun Chou]
    > - Dept: Department of Management Information Systems, National Chengchi University
    > 
    > - Contacts: chuang@nccu.edu.tw
- [Dr. Yen-Chun Chou]
    > - Dept: Department of Management Information Systems, National Chengchi University
    > 
    > - Contacts: yenchun@nccu.edu.tw
- [Dr. Shun-Wen Hsiao]
    > - Dept: Department of Management Information Systems, National Chengchi University
    > 
    > - Contacts: hsiaom@nccu.edu.tw
- [Dr. Chin-Sheng Yang]
    > - Dept: Department of Information Management, Yuan Ze University
    > 
    > - Contacts: csyang@saturn.yzu.edu.tw

### Thesis Author
- Jiun-Yi Yang
    > - Dept: Department of Information Management, Yuan Ze University
    > 
    > - Contacts: 108356035@nccu.edu.tw

### Research Process
![image](https://i.imgur.com/qyRw0As.jpg)
![image](https://i.imgur.com/nYhv2Jf.jpg)

### Script Explanation
1. 以分析案例區分資料夾，分別存放資料集、處理後產物、分群紀錄
2. 各檔案用途說明如下：
    - Cianiao/  
        - Cainiao_dataset/   
            > *菜鳥網絡部分資料集*
        - Cainiao_preprocessed/   
            > *處理後的狀態序列及其他衍生檔案*
        - Cainiao_output/   
            > *分群後標籤及統計指標結果檔*
        - cainiao-clusterMetrics.py
            > *計算分群標籤移轉比例、統計指標*
        - cainiao-makeSequence.py 
            > *對訂單抽樣、將原始日誌資料轉換為狀態序列*
    - Tmall/
        - Tmall_dataset/
            > *天貓用戶行為日誌資料*
        - Tmall_preprocessed/
            > *前處理後資料集、處理後的狀態序列及其他衍生檔案*
        - Tmall_output/
            > *分群後標籤及統計指標結果檔*
        - tmall-dataPreprocessing.py
            > *對原始資料集前處理，如動作欄位類別轉換等*
        - tmall-clusterMetrics.py
            > *計算分群標籤移轉比例、統計指標*
        - tmall-makeSequence.py   
            > *狀態定義、將原始日誌資料轉換為狀態序列*
    - optimalMatchingAnalysis.R
        > *計算相異度矩陣、分群並繪製序列狀態比例分布圖*

3. 檔案運行順序：

- Tmall: `tmall-dataPreprocessing.py` &#8594; `tmall-makeSequence.py` &#8594; `optimalMatchingAnalysis.R` &#8594; `tmall-clusterMetrics.py`
- Cianiao: `cianiao-makeSequence.py` &#8594; `optimalMatchingAnalysis.R` &#8594; `cianiao-clusterMetrics.py`


### Related Links
Since the storage limit, datasets are stored on this [Google Drive link]

大型資料檔已另外存放於此 [Google Drive 雲端資料夾]

---

If you're interested in the project, and need folder permission, please contact me at [jiunyi.yang.abao@gmail.com]


<!-- superlink reference -->
[Dr. Hao-Chun Chou]: https://mis2.nccu.edu.tw/en/Faculty/Faculty_01/%E8%8E%8A-%E7%9A%93%E9%88%9E-68290166
[Dr. Yen-Chun Chou]: https://mis2.nccu.edu.tw/en/Faculty/Faculty_01/YEN-CHUN-CHOU-53848898
[Dr. Shun-Wen Hsiao]: https://mis2.nccu.edu.tw/en/Faculty/Faculty_01/Shun-Wen-Hsiao-19738494
[Dr. Chin-Sheng Yang]: https://www.mis.yzu.edu.tw/teacher_profile.aspx?lang=cht&pid=1075&tchid=022&leng=en
[Google Drive 雲端資料夾]: https://drive.google.com/drive/folders/1xNEHPtwk44GFA31XsaXaHlXCbkUUqMmm?usp=sharing
[Google Drive link]: https://drive.google.com/drive/folders/1xNEHPtwk44GFA31XsaXaHlXCbkUUqMmm?usp=sharing
[jiunyi.yang.abao@gmail.com]: jiunyi.yang.abao@gmail.com
