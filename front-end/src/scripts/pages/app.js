import routes from "../routes/routes.js";
import { getActiveRoute } from "../routes/url-parser.js";

class App {
  #content = null;
  #hamburgerMenu = null;
  #mobileNavigation = null;

  constructor({ content, hamburgerMenu, mobileNavigation }) {
    this.#content = content;
    this.#hamburgerMenu = hamburgerMenu;
    this.#mobileNavigation = mobileNavigation 

    this.#init();
  }

  #init() {
    this._setupHamburgerMenu();
  }

  _setupHamburgerMenu() {
    this.#hamburgerMenu.addEventListener('click', (event) => {
      event.stopPropagation();
      this.#mobileNavigation.classList.toggle('active');
    })
  }

  async renderPage() {
    const url = getActiveRoute();
    const page = routes[url];

    this.#content.innerHTML = await page.render();
    await page.afterRender();
  }
}

export default App;