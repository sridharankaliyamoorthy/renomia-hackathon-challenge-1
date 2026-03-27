import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'MrRooT.Ai — Renomia Challenge 1',
  tagline: 'Insurance offer comparison — architecture & operations',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // For GitHub Pages (project site), set e.g. url: 'https://USER.github.io' and baseUrl: '/REPO/'
  url: 'https://sridharankaliyamoorthy.github.io',
  baseUrl: '/',

  organizationName: 'sridharankaliyamoorthy',
  projectName: 'renomia-hackathon-challenge-1',

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/sridharankaliyamoorthy/renomia-hackathon-challenge-1/tree/main/documentation/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Renomia Challenge 1',
      logo: {
        alt: 'Documentation',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          href: 'https://github.com/sridharankaliyamoorthy/renomia-hackathon-challenge-1',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Overview', to: '/docs/intro'},
            {label: 'Architecture', to: '/docs/architecture'},
            {label: 'Operations', to: '/docs/operations'},
          ],
        },
        {
          title: 'Repository',
          items: [
            {
              label: 'renomia-hackathon-challenge-1',
              href: 'https://github.com/sridharankaliyamoorthy/renomia-hackathon-challenge-1',
            },
          ],
        },
      ],
      copyright: `MrRooT.Ai · Renomia Hackathon · ${new Date().getFullYear()} · Built with Docusaurus`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
