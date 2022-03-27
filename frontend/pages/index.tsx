import type { NextPage } from "next";
import Layout from "../components/layout";
import Dropzone from "../components/dropzone";
import { Flex, Heading, Stack } from "@chakra-ui/react";
import Prediction from "../components/prediction";

const Home: NextPage = () => {
  return (
    <Layout>
      <Stack pt={6} direction="column" textAlign={"center"} gap={2}>
        <Heading color="gray.600" as={"h1"}>
          12-lead ECG Classification
        </Heading>
        <Dropzone />
        <Prediction normality={0.77} lvefgteq40={0.6} lveflw50={0.3} />
      </Stack>
    </Layout>
  );
};

export default Home;
