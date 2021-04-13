# Image Stitching

## How to Use

## Build

### Required

* opencv
  * 安裝後將 `opencv` 底下的 `\build\x64\vc15\bin` 的路徑加入

## How to Build

* ``` bash
  git submodule add https://github.com/jlblancoc/nanoflann.git ./ImageStitching/Dependency/nanoflann
  ```
* 在 Visual Studio 中，對專案名稱點右鍵，選擇`屬性`
  * 在屬性頁面中，將組態設為`所有組態`、平台設為`所有平台`
  * 選擇 `VC++ 目錄`
    * 修改 `Include 目錄` 中 Library 的路徑
    * 修改 `程式庫目錄` 中 Library 的路徑
    * 
* 即可進行編譯與執行程式

## Reference

* [Homework 2 Image Stitching](https://www.csie.ntu.edu.tw/~b97074/vfx_html/hw2.html)
* [Harris Corner Detection](https://blog.csdn.net/u014485485/article/details/79056666)
* [A Combined Corner and Edge Detector](https://www.ece.lsu.edu/gunturk/EE7700/A_Combined_Corner_and_Edge_Detector.pdf)
* [A Combined Corner and Edge Detector 論文翻譯](https://blog.csdn.net/baidu_37336262/article/details/110123728)
