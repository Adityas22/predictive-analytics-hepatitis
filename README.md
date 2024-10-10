# Proyek Pertama: Hepatitis Predict
Disusun oleh: Aditya Septiawan

## Domain Proyek
Domain yang dipilih untuk proyek machine learning ini adalah sosial, dengan judul "Perbadingan Algortima dalam Memprediksi Penyakit Hepatitis C"

#### Latar Belakang
<p align="center">
  <img src="https://github.com/Adityas22/predictive-analytics-hepatitis/blob/main/image/kasus%20hepatitis.png" alt="docs12" width="470">
</p>
<br>
<p align="justify">
Hepatitis C adalah penyakit infeksi yang menyerang hati akibat virus hepatitis C (HCV) dan dapat berkembang menjadi kondisi serius seperti fibrosis, sirosis, serta kanker hati jika tidak dideteksi dan diobati dengan baik. Penyakit ini sering kali tidak menunjukkan gejala pada tahap awal, sehingga banyak penderita tidak menyadari bahwa mereka terinfeksi hingga terjadi kerusakan hati yang signifikan. Berdasarkan laporan global dari Organisasi Kesehatan Dunia (WHO), diperkirakan 58 juta orang di seluruh dunia menderita infeksi hepatitis C kronis, dengan sekitar 1,5 juta infeksi baru setiap tahunnya. Selain itu, terdapat sekitar 3,2 juta remaja dan anak-anak yang juga terinfeksi hepatitis C kronis. Pada tahun 2019, WHO mencatat bahwa sekitar 290.000 orang meninggal akibat komplikasi hepatitis C, terutama karena sirosis dan kanker hati primer (karsinoma hepatoseluler) <a href="https://journal.stmikjayakarta.ac.id/index.php/JMIJayakarta/article/view/1098/732">[1]</a>. Tes diagnostik untuk mengidentifikasi orang dengan Hepatitis C dan tahap perkembangan penyakitnya (seperti fibrosis dan sirosis) bisa sangat membantu. Dengan pendekatan ini, deteksi dan identifikasi yang lebih akurat dapat dilakukan untuk memantau kondisi pasien dan memberikan informasi yang lebih tepat mengenai status penyakit.
</p>

## Business Understanding

#### Problem Statements
Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:
- Bagaimana membuat model machine learning yang dapat memprediksi atau mendiagnosis virus hepatitis C (HCV) pada pasien berdasarkan data demografi dan hasil laboratorium?
- Model yang seperti apa yang memiliki akurasi paling baik untuk diagnosis tersebut?

#### Goals
Tujuan dari proyek ini adalah:
- Membuat model machine learning yang dapat memprediksikan pasien apakah terdiagnosis virus hepatitis C (HCV) atau tidak, berdasarkan data demografi dan hasil laboratorium.
- Membandingkan beberapa algoritma model sehingga ditemukan akurasi yang paling baik untuk memprediksikan diagnosis virus hepatitis C (HCV) berdasarkan data demografi dan hasil laboratorium.

#### Solution statements
Untuk mencapai tujuan tersebut, dalam proyek ini akan dibuat beberapa model yang berbeda untuk dibandingkan, diantaranya adalah menggunakan:
- <p align="justify"> K-Nearest Neighbor (KNN) adalah algoritma yang sederhana dan efisien yang digunakan untuk mengklasifikasikan data baru berdasarkan kesamaan dengan data yang sudah ada. Algoritma ini bekerja dengan cara mencari titik data terdekat (tetangga) dalam dataset pelatihan dan mengklasifikasikan data baru berdasarkan mayoritas kelas dari tetangga tersebut <a href="https://www.geeksforgeeks.org/k-nearest-neighbours/">[2]</a>. </p>
- <p align="justify"> Algoritma Support Vector Machine (SVM) digunakan untuk menemukan sebuah hyperplane dalam ruang N-dimensi (di mana N merupakan jumlah fitur) yang secara efektif mengklasifikasikan titik-titik data. SVM dapat digunakan untuk menyelesaikan masalah-masalah klasifikasi, regresi, serta deteksi outlier <a href="https://www.geeksforgeeks.org/support-vector-machine-algorithm/">[3]</a>. </p>
- <p align="justify"> Random Forest adalah algoritma pembelajaran mesin yang kuat dan fleksibel, digunakan untuk berbagai tugas seperti klasifikasi dan regresi. Sebagai metode ensemble, Random Forest terdiri dari banyak pohon keputusan kecil, yang dikenal sebagai estimator, di mana masing-masing menghasilkan prediksi independen. Algoritma ini menggabungkan hasil dari semua estimator untuk menghasilkan prediksi yang lebih akurat <a href="https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/">[4]</a>. </p>
- <p align="justify"> Naive Bayes adalah model pembelajaran mesin yang bersifat probabilistik dan digunakan untuk tugas klasifikasi. Inti dari pengklasifikasi ini didasarkan pada teorema Bayes, yang memungkinkan perhitungan probabilitas untuk mengklasifikasikan data. Naive Bayes mengasumsikan bahwa setiap fitur dalam dataset bersifat independen satu sama lain, yang menyederhanakan proses klasifikasi <a href="https://www.geeksforgeeks.org/naive-bayes-classifiers//">[5]</a>. </p>

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah data hasil sampel pasien yaitu sebanyak 615 sampel. Data ini dapat diunduh melalui UCI Machine Learning Repository. Pada dataset ini terdapat 14 kolom, diantaranya:
1.**Unnamed: 0** : Nomor urut pasien pada file CSV, hanya digunakan untuk keperluan identifikasi internal dalam dataset dan tidak berperan dalam analisis model.
2. **Category** : Kategori diagnosis pasien, yang menunjukkan status kesehatan terkait Hepatitis C:
   - 0 = Blood Donor (Pendonor darah)
   - 0s = Suspect Blood Donor (Pendonor darah yang dicurigai)
   - 1 = Hepatitis
   - 2 = Fibrosis (kerusakan hati yang menyebabkan jaringan parut)
   - 3 = Cirrhosis (kerusakan hati kronis dengan jaringan parut parah)
3. **Age** : Usia pasien dalam tahun.
4. **Sex** : Jenis kelamin pasien, dengan nilai 'f' untuk perempuan dan 'm' untuk laki-laki.
5. **ALB** (Albumin): Protein utama dalam darah yang diproduksi oleh hati, penting untuk menjaga tekanan osmotik darah.
6. **ALP** (Alkaline Phosphatase): Enzim yang ditemukan di hati, tulang, dan jaringan lain, sering digunakan sebagai indikator kesehatan hati dan tulang.
7. **ALT** (Alanine Aminotransferase): Enzim hati yang dilepaskan ke darah saat terjadi kerusakan pada sel hati.
8. **AST** (Aspartate Aminotransferase): Enzim yang terdapat di hati dan organ lain, meningkat saat ada kerusakan pada hati atau jantung.
9. **BIL** (Bilirubin): Produk pemecahan sel darah merah, kadar yang tinggi bisa menunjukkan kerusakan hati atau masalah dengan saluran empedu.
10. **CHE** (Cholinesterase): Enzim yang diproduksi oleh hati, digunakan untuk menilai fungsi hati dan kondisi kesehatan umum.
11. **CHOL** (Cholesterol): Kadar kolesterol total dalam darah, yang dapat dipengaruhi oleh fungsi hati.
12. **CREA** (Creatinine): Produk limbah yang dihasilkan oleh otot, digunakan untuk menilai fungsi ginjal.\
13. **GGT** (Gamma-Glutamyl Transferase): Enzim yang meningkat pada penyakit hati, terutama pada penyalahgunaan alkohol atau kerusakan saluran empedu.
14. **PROT** (Total Protein): Jumlah total protein dalam darah, termasuk albumin dan globulin, yang berfungsi sebagai indikator status nutrisi dan fungsi hati.

     gambar
#### Informasi Dataset:

<table>
  <tr>
    <th>Jenis</th>
    <th>Keterangan</th>
  </tr>
  <tr>
    <td>Title</td>
    <td>HCV data</td>
  </tr>
  <tr>
    <td>Source</td>
    <td> <a href="https://archive.ics.uci.edu/dataset/571/hcv+data">UCI Machine Learning Repository</a></td>
  </tr>
  <tr>
    <td>Creators</td>
    <td>Ralf Lichtinghagen, Frank Klawonn, Georg Hoffmann</td>
  </tr>
  <tr>
    <td>License</td>
    <td>Creative Commons Attribution 4.0 International (CC BY 4.0)</td>
  </tr>
  <tr>
    <td>Characteristics</td>
    <td>Multivariate</td>
  </tr>
  <tr>
    <td>Subject Area</td>
    <td>Health and Medicinee</td>
  </tr>
</table>

<br>
=====

##### Gambar Korelasi

<p align="center">
  <img src="https://github.com/Adityas22/predictive-analytics-hepatitis/blob/main/image/korelasi.png" alt="docs12" width="470">
</p>

#### Link Referensi
[Link Referensi Artikel](https://ejournal.nusamandiri.ac.id/index.php/pilar/article/view/149/126)

#### Sitasi
<p align="justify">
Hepatitis amat sangat bahaya [1](https://ejournal.nusamandiri.ac.id/index.php/pilar/article/view/149/126).
</p>

## References
[1]: https://journal.stmikjayakarta.ac.id/index.php/JMIJayakarta/article/view/1098/732
