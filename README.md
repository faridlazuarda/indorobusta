# Tugas-Akhir-Repository

### Findings:
- Ternyata metode ini berhasil untuk mengelabui model IndoBERT, walaupun tidak meminimalkan jumlah perturbasi, namun dapat digunakan menjadi pengembangan lebih efektif dari EDA (Easy Data augmentation) tanpa merubah makna semantik secara signifikan
- Universal Sentence Encoder ternyata ada untuk multilingual, tapi ga spesifik untuk code mixing, Cuma cross lingual
    - Kuncinya ada disini: https://blog.gdeltproject.org/experiments-using-universal-sentence-encoder-embeddings-for-news-similarity/
    - Karena ternyata multilingual USE belum perfect2 amat, jadinya pake yg paling cepat aja yaitu monolingual USE v4 menggunakan DAN (Dynamic Average Network)

### Catch:
- Karena pakai perturbation probability, jadinya bukan adversarial attack karena adversarial attack meminimumkan perturbasi dan memaksimalkan efek. Kalo ini tetap memaksimumkan efek, tapi masih menggunakan probability untuk memilih jumlah perturbasinya.
- USE ini ditrain menggunakan monolingual corpora, belum tau apakah cukup robust kalo dipake di bahasa lain terutama ini mixing dari 2 bahasa
- Catch lain tentang USE ada di findings

### Issues:
- Gimana cara menemukan list of words_perturb dengan perubahan makna semantik terkecil (menemukan equilibrium dari tradeoff effective perturbation - semantic similarity) (solved)
    * Solusi 1: kurangi jumlah kata yang diperturb. Kalo udah last gimana? Bener jg
    * Solusi 2: Coba ganti least important word di most important word dengan kandidat lain
    * Solusi 3: cari subset permutation dari list of words_perturb, terus bikin semantic similarity dan effectivity dictionary, terus sort by {effectivity, similarity} -> dengan memperhatikan threshold
        1. Gimana kalo tetep nggak memenuhi similarity? Lakukan cara yg sama tapi dengan mengurangi jumlah perturbasi kata. 
            * Gimana kalo tetep ngga memenuhi kriteria sampe akhir? Berarti attack gagal dan akan dikembalikan spt teks semula
        2. Bikin map:
            * {kalimat kandidat perturbasi:(similarity score, importance score)} terus disort berdsasarkan similarity terus importance

### Experiment parameters:
1. seed
2. Target model
3. Attack type (word importance/random)
4. Codemixing/synonym replacement
5. Perturbation ratio
6. Jumlah data
7. Downstream_task

### To-Do next:
1. optimization
2. Lengkapi modul yg belum selesai (synonym attack, attack on emotion classification task)
3. Eval before and after attack accuracy
4. Run experiment on the whole dataset
5. Logging/pencatatan eksperimen
