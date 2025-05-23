
export default class CropRecommendationPage {
  #takenDocumentation = null;

  async render() {
    return `
      <div class="main-feature-content">
        <div class="jumbotron">
          <img src="images/crop-recommendation.jpeg" alt="">
          <h1 class="title">Crop <span>Recommendation</span></h1>
        </div>
        
        <div class="container">
          <div class="description">
            <p>Pengguna dapat memperoleh rekomendasi tanaman yang paling sesuai untuk ditanam dengan memasukkan informasi kondisi tanah dan cuaca di sekitarnya. Sistem berbasis AI akan menganalisis data tersebut dan menampilkan daftar tanaman yang direkomendasikan, lengkap dengan tingkat kecocokan, kebutuhan dasar, serta tips awal untuk memulai penanaman secara efektif.</p>
          </div>

          <div class="recomendation">
            <div class="top-content">
              <i class="bi bi-search"></i>
              <p>Rekomendasi</p>
            </div>

            <div class="bottom-content main-recommendation-content">
              <div class="item">
                <p class="title">🔍 Data yang Digunakan</p>
                <p>Sistem ini menggunakan data seperti kandungan Nitrogen (N), Fosfor (P), Kalium (K), suhu, kelembapan, pH tanah, dan curah hujan untuk memberikan rekomendasi tanaman yang paling sesuai dengan kondisi lingkungan Anda.</p>
              </div>
              <div class="item">
                <p class="title">🤖 Keunggulan Teknologi yang Digunakan</p>
                <p>Didukung oleh model machine learning yang dilatih menggunakan dataset pertanian, sistem ini mampu memberikan rekomendasi tanaman dengan akurasi tinggi berdasarkan input numerik.</p>
              </div>
              <div class="item">
                <p class="title">📂 Cara Menggunakan Sistem Ini</p>
                <ul>
                  <li>Unggah file CSV berisi parameter lingkungan (seperti contoh: N, P, K, suhu, dll).</li>
                  <li>Sistem akan menganalisis data dan merekomendasikan jenis tanaman yang optimal.</li>
                  <li>Tidak memerlukan gambar atau proses pemindaian visual tanaman.</li>
                </ul>
              </div>
              <div class="item">
                <form id="form-upload">
                  <div class="upload-button-container">
                    <label for="image-file" class="btn file-button upload-file">Unggah File</label>
                    <input type="file" id="image-file" name="image-file" style="display: none">
                  </div>
                  <div class="image-preview" id="image-preview"></div>
                  <div class="detection-button-container" id="detection-button-container">
                    <button type="button" class="detection-button"> Deteksi Sekarang</button>
                  </div>
                </form>
              </div>
              <div class="item" id="output">
                <p class="title output-title"></p>
                <p class="output-desc"></p>
              </div>
              <div class="item" id="suggestion">
                <p class="title suggestion-title"></p>
                <p class="suggestion-desc"></p>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  async afterRender() {
    this.#takenDocumentation = null;
    this.#setupForm();
  }

  #setupForm() {
    const imageFile = document.querySelector('#image-file');
    const imagePreview = document.querySelector('#image-preview');

    imageFile ? imageFile.addEventListener('change', () => {
      const file = imageFile.files[0];

      if (!file) return;

      this.#takenDocumentation = {
        id: `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
        blob: file,
        url: URL.createObjectURL(file),
        type: file.type,
        name: file.name
      };

      const picture = this.#takenDocumentation;

      let html = '';

      if (picture.type.startsWith('image/')) {
        html = `
          <button type="button" data-deletepictureid="${picture.id}" class="new-form__documentations__outputs-item__delete-btn">
            <img src="${picture.url}" alt="preview">
          </button>
        `;
      } else {
        html = `
          <button type="button" data-deletepictureid="${picture.id}" class="new-form__documentations__outputs-item__delete-btn">
            <div class="file-preview">
              <i class="bi bi-file-earmark-text"></i>
              <p>${picture.name}</p>
            </div>
          </button>
        `;
      }

      imagePreview.innerHTML = '';
      imagePreview.innerHTML = html;

      document.querySelector('button[data-deletepictureid]')?.addEventListener('click', (event) => {
        const pictureId = event.currentTarget.dataset.deletepictureid;
        const deleted = this.#removePicture(pictureId);
        if (!deleted) {
          console.log(`Picture with id ${pictureId} was not found`);
        }
        this.#populateTakenPictures();

        const detectionButton = document.querySelector('.detection-button');
        detectionButton ? detectionButton.classList.remove('active') : null;
        
        const output = document.querySelector('#output');
        const suggestion = document.querySelector('#suggestion');

        output ? output.classList.remove('active') : null;
        suggestion ? suggestion.classList.remove('active') : null;
      });
      
      const detectionButton = document.querySelector('.detection-button');
      detectionButton ? detectionButton.classList.add('active') : null;

      this.#setupDecationButton();
    }) : null
  }

  async #populateTakenPictures() {
    const imagePreview = document.querySelector('#image-preview');

    if (!this.#takenDocumentation) {
      imagePreview.innerHTML = '';
      return;
    }

    const picture = this.#takenDocumentation;
    const imageUrl = URL.createObjectURL(picture.blob);

    let html = '';

    if (picture.type.startsWith('image/')) {
      html = `
        <button type="button" data-deletepictureid="${picture.id}" class="new-form__documentations__outputs-item__delete-btn">
          <img src="${imageUrl}" alt="preview">
        </button>
      `;
    } else {
      html = `
        <button type="button" data-deletepictureid="${picture.id}" class="new-form__documentations__outputs-item__delete-btn">
          <div class="file-preview">
            <i class="bi bi-file-earmark-text"></i>
            <p>${picture.name}</p>
          </div>
        </button>
      `;
    }

    imagePreview.innerHTML = html;

    document.querySelector('button[data-deletepictureid]')?.addEventListener('click', (event) => {
      const pictureId = event.currentTarget.dataset.deletepictureid;
      const deleted = this.#removePicture(pictureId);
      if (!deleted) {
        console.log(`Picture with id ${pictureId} was not found`);
      }
      this.#populateTakenPictures();

      const detectionButton = document.querySelector('.detection-button');
      detectionButton ? detectionButton.classList.remove('active') : null;

      const output = document.querySelector('#output');
      const suggestion = document.querySelector('#suggestion');

      output ? output.classList.remove('active') : null;
      suggestion ? suggestion.classList.remove('active') : null;
    });
  }

  #removePicture(id) {
    if (this.#takenDocumentation && this.#takenDocumentation.id === id) {
      const deleted = this.#takenDocumentation;
      this.#takenDocumentation = null;
      return deleted;
    }

    return null;
  }

  #setupDecationButton() {
    const detectionButton = document.querySelector('.detection-button');
    detectionButton ? detectionButton.addEventListener('click', (event) => {
      event.stopPropagation();
      
      const output = document.querySelector('#output');
      const suggestion = document.querySelector('#suggestion');

      output ? output.classList.add('active') : null;
      suggestion ? suggestion.classList.add('active') : null;

      const outputTitle = document.querySelector('.output-title');
      const outputDesc = document.querySelector('.output-desc');
      const suggestionTitle = document.querySelector('.suggestion-title');
      const suggestionDesc = document.querySelector('.suggestion-desc');
 
      outputTitle ? outputTitle.textContent = '' : null;
      outputTitle ? outputTitle.textContent = '🔍 Hasil' : null;
      outputDesc ? outputDesc.textContent = '' : null;
      outputDesc ? outputDesc.textContent = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut ultricies posuere ipsum eget fermentum. Maecenas sodales urna eu accumsan sodales. Duis non ipsum mollis, feugiat sem tempor, dapibus magna.' : null;

      suggestionTitle ? suggestionTitle.textContent = '' : null;
      suggestionTitle ? suggestionTitle.textContent = '💡 Saran' : null;
      suggestionDesc ? suggestionDesc.textContent = '' : null;
      suggestionDesc ? suggestionDesc.textContent = 'Praesent nec consectetur neque, vel efficitur neque. Vestibulum non turpis a nisl ornare imperdiet vitae et nisi. Duis tincidunt lobortis tellus ac viverra. Sed suscipit varius imperdiet. Ut condimentum pretium odio, non blandit metus viverra id. Proin sollicitudin metus nec volutpat commodo.' : null;

    }) : null;
  }

};