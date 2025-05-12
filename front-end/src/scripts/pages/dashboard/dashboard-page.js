export default class DashboardPage {
  async render() {
    return `<div class="dashboard-content">
      <div id="carouselExample" class="carousel carousel-dark slide">
        <div class="carousel-indicators">
          <button type="button" data-bs-target="#carouselExample" data-bs-slide-to="0" class="active" aria-current="true" aria-label="Slide 1"></button>
          <button type="button" data-bs-target="#carouselExample" data-bs-slide-to="1" aria-label="Slide 2"></button>
          <button type="button" data-bs-target="#carouselExample" data-bs-slide-to="2" aria-label="Slide 3"></button>
        </div>
        <div class="carousel-inner">
          <div class="carousel-item active" data-bs-interval="10000">
            <img src="images/carousel-1.jpg" class="d-block w-100" style="" alt="...">
          </div>
          <div class="carousel-item" data-bs-interval="2000">
            <img src="images/carousel-1.jpg" class="d-block w-100" style="" alt="...">
          </div>
          <div class="carousel-item">
            <img src="images/carousel-1.jpg" class="d-block w-100" style="" alt="...">
          </div>
        </div>
        <button class="carousel-control-prev" type="button" data-bs-target="#carouselExample" data-bs-slide="prev">
          <i class="bi bi-chevron-left" style="padding: 10px; background-color: rgba(255, 255, 255, .3) !important; font-size: 25px; color: #000;"></i>
        </button>
        <button class="carousel-control-next" type="button" data-bs-target="#carouselExample" data-bs-slide="next">
          <i class="bi bi-chevron-right" style="padding: 10px; background-color: rgba(255, 255, 255, .3) !important; font-size: 25px; color: #000;"></i>
        </button>
      </div>
    </div>`
  }

  async afterRender() {
    // Add any additional functionality or event listeners here
    
  }
}