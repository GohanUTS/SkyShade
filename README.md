# SkyShade Website

This branch contains the GitHub Pages website for the SkyShade AI Robotics project.

Live website:

```text
https://gohanuts.github.io/SkyShade/
```

The website presents a clear summary of the project, including:

- Project overview
- System architecture
- Running the simulation
- Sub-1: Perception
- Sub-2: Flight Control
- Sub-3: Environmental Decision
- Sub-4: Navigation Safety
- Validation and results
- Known issues and development decisions

The website is based on the Nerfies website template and has been adapted for the SkyShade ROS2 and PyBullet robotics simulation project.

## Website Files

The main homepage file is:

```text
index.html
```

The supporting website assets are stored in:

```text
static/
```

The project documentation copied from the main branch is stored in:

```text
wiki/
```

## Preview Locally

To preview the website locally, run:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## Updating the Website

Make sure you are on the website branch:

```bash
git checkout github_page
```

After editing the website files, commit and push:

```bash
git add index.html README.md static wiki
git commit -m "Update SkyShade website"
git push origin github_page
```

The live website should update shortly after pushing.

## Updating Wiki Content

The latest wiki content is maintained on the main branch. To copy the latest wiki into this website branch:

```bash
git checkout github_page
git checkout main -- wiki
git add wiki
git commit -m "Update website wiki content"
git push origin github_page
```

Only update `index.html` if the homepage summary, layout, images, or links need to change. If only the detailed wiki text changes, updating the `wiki/` folder is enough.

## GitHub Pages Setup

This branch is used as the GitHub Pages source branch.

GitHub Pages should be configured as:

```text
Settings → Pages → Deploy from a branch → github_page → /root
```

## Credits

Website template adapted from:

```text
https://github.com/nerfies/nerfies.github.io
```