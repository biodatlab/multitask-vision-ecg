import { ChakraProvider, extendTheme } from "@chakra-ui/react";
import "@fontsource/ibm-plex-sans-thai";
import "@fontsource/ibm-plex-sans-thai/400.css";
import "@fontsource/ibm-plex-sans-thai/600.css";
import type { AppProps } from "next/app";
import Head from "next/head";
import { MirageProvider } from "../contexts/MirageContext";
import "../styles/globals.css";

const theme = extendTheme({
  fonts: {
    heading: '"IBM Plex Sans Thai", sans-serif',
    body: '"IBM Plex Sans Thai", sans-serif',
  },
  colors: {
    primary: {
      50: "#FFF2EF",
      100: "#FFBBB1",
      200: "#FF8E7F",
      300: "#FF624D",
      400: "#FE361B",
      500: "#E51E01",
      600: "#B31600",
      700: "#850D00",
      800: "#6F0800",
      900: "#4F0600",
    },
    secondary: {
      50: "#F9FAFF",
      100: "#C9D1EC",
      200: "#A7B3D9",
      300: "#8494C7",
      400: "#6375B6",
      500: "#495C9C",
      600: "#38477B",
      700: "#3B466F",
      800: "#283359",
      900: "#151F38",
    },
    gray: {
      50: "#F6F9FC",
      100: "#EAF0F6",
      200: "#DEE5EE",
      300: "#C4CFDC",
      400: "#96A5B8",
      500: "#66758B",
      600: "#414B5D",
      700: "#28303F",
      800: "#191D27",
      900: "#161820",
    },
  },
  styles: {
    global: {
      body: {
        color: "gray.800",
      },
      p: {
        color: "gray.800",
      },
    },
  },
});

function MyApp({ Component, pageProps }: AppProps) {
  // prettier-ignore
  return (
    <MirageProvider>
      <Head>
        <link rel="apple-touch-icon" sizes="180x180" href="/favicon/apple-touch-icon.png" />
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon/favicon-32x32.png" />
        <link rel="icon" type="image/png" sizes="16x16" href="/favicon/favicon-16x16.png" />
        <link rel="manifest" href="/favicon/site.webmanifest" />
        <link rel="mask-icon" href="/favicon/safari-pinned-tab.svg" color="#ed64a6" />
        <link rel="shortcut icon" href="/favicon/favicon.ico" />
        <meta name="msapplication-TileColor" content="#ed64a6" />
        <meta name="msapplication-config" content="/favicon/browserconfig.xml" />
        <meta name="theme-color" content="#ffffff" />
      </Head>
      <ChakraProvider theme={theme}>
        <Component {...pageProps} />
      </ChakraProvider>
    </MirageProvider>
  );
}

export default MyApp;
