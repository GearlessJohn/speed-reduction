<!-- PROJECT LOGO -->
<div align="center">
  <h3 align="center">Speed Reduction</h3>
  <p align="center">
    A full assessment of the impacts of vessel speed reduction.
    <br />
    <a href="https://github.com/GearlessJohn/speed-reduction"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/GearlessJohn/speed-reduction">View Demo</a>
    ·
    <a href="https://github.com/GearlessJohn/speed-reduction/issues">Report Bug</a>
    ·
    <a href="https://github.com/GearlessJohn/speed-reduction/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

This project aims to provide an all-encompassing analysis of the consequences of global vessel speed reduction.
Considering the current carbon tax and IMO policies, the impacts are mainly analyzed for the years 2023-2026.

The analytical aspects include:

* Fleet income
* Greenhouse gas emissions
* Shipbuilding and market supply/demand balance

Use the `READEME.md` to get started.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

The whole project is built on Python. 
All the modules used are included in the `requirement.txt`, file.
Use the following command to install all required packages:
* powershell/cmd/bash
  ```sh
  pip install -r requirements.txt
  ```

[//]: # (### Installation)

[//]: # ()
[//]: # (_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't)

[//]: # (rely on any external dependencies or services._)

[//]: # ()
[//]: # (1. Clone the repo)

[//]: # (   ```sh)

[//]: # (   git clone https://github.com/your_username_/Project-Name.git)

[//]: # (   ```)

[//]: # (2. Install NPM packages)

[//]: # (   ```sh)

[//]: # (   npm install)

[//]: # (   ```)

[//]: # (3. Enter your API in `config.js`)

[//]: # (   ```js)

[//]: # (   const API_KEY = 'ENTER YOUR API';)

[//]: # (   ```)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->

## Usage

The project will automatically read the vessel information stored in the `./data/CACIB-SAMPLE.xlsx`. 
All simulation scenarios are encapsulated in the `./sample/main.py` file.
Go to the project root directory and use the following command in cmd/powershell/bash to launch every scenario:

* Step 1-Optimization of annual profit per vessel: 
  ```sh
  python .\sample\__main__.py 0
  ```
* Step 2-Optimization of 4-year profit for a single vessel (including CII limitations):
  ```sh
  python .\sample\__main__.py 3
  ```
* Step 3-Optimization of 4-year profit for a fleet (with shipbuilding):
  ```sh
  python .\sample\__main__.py 5
  ```
* Step 4-Optimization of 4-year profit for a fleet with flexible prices :
  ```sh
  python .\sample\__main__.py 6
  ```
* Mean-field model:
  ```sh
  python .\sample\__main__.py 2
  ```
_For more information of each simulation step, please refer to the [Documentation](https://github.com/GearlessJohn/speed-reduction/tree/main/report)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->

## Roadmap

- [x] Annual profit optimization
- [x] 4-year profit optimization
- [x] Add shipbuilding
- [x] Apply flexible fret rates
- [ ] Fret rates forecasting
- [ ] Accurate price elasticity

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->

## Contact

Xuan WANG - xuan.wang@ca-cib.com

Project Link: [https://github.com/GearlessJohn/speed-reduction](https://github.com/GearlessJohn/speed-reduction)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites
to kick things off!

* [An improved ship speed optimization method](https://www.sciencedirect.com/science/article/pii/S0959652622053884)
* [Cost assessment of alternative fuels](https://www.sciencedirect.com/science/article/pii/S1361920922002425)
* [A short course on Mean field games](https://www.ceremade.dauphine.fr/~cardaliaguet/MFGcours2018.pdf)
* [Environmental economic analysis of speed reduction](https://link.springer.com/article/10.1007/s11356-023-26745-4)
* [Incorporating Logistics in Freight Transport](https://www.researchgate.net/publication/233482037_Incorporating_Logistics_in_Freight_Transport_Demand_Models_State-of-the-Art_and_Research_Opportunities)
* [Applications of Life Cycle Assessment in Shipping](https://www.researchgate.net/publication/280313533_Applications_of_Life_Cycle_Assessment_in_Shipping)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
