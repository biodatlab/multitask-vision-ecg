import type { NextPage } from "next";
import Layout from "../components/layout";
import Dropzone from "../components/dropzone";
import { Divider, Heading, Stack, Text } from "@chakra-ui/react";
import Prediction from "../components/prediction";
import { useState } from "react";

const Home: NextPage = () => {
  const [result, setResult] = useState(null);
  const [isLoadingResult, setIsLoadingResult] = useState(false);

  return (
    <Layout>
      <Stack py={6} direction="column" textAlign={"center"} gap={2}>
        <Heading color="gray.500" as={"h1"}>
          12-lead ECG Classification
        </Heading>
        <Text textAlign="left">
          <Text as="span" fontWeight="bold">
            หทัย AI
          </Text>{" "}
          เป็นแอพพลิเคชั่นที่ใช้ AI ในการทำนายผลของภาพสแกนคลื่นไฟฟ้าหัวใจ
          (Electrocardiogram, ECG) แบบ 12 Lead ในรูปแบบของภาพหรือไฟล์ PDF
          โดยโมเดลจะทำนายความน่าจะเป็นของการมีรอยแผลเป็นในหัวใจ (Myocardial
          scar) และค่าประสิทธิภาพการทำงานของหัวใจห้องล่างซ้าย (Left ventricular
          ejection fraction, LVEF)
        </Text>
        <Dropzone
          onClearFile={() => setResult(null)}
          onSubmit={(f) => {
            setIsLoadingResult(true);
            console.log(f);

            const formdata = new FormData();
            formdata.append("file", f, f.name);

            fetch("http://localhost:8000/api/predict", {
              method: "POST",
              body: formdata,
            })
              .then((res) => {
                return res.json();
              })
              .then((resJson) => {
                setResult(resJson);
                console.log("resJson", resJson);
              })
              .finally(() => {
                setIsLoadingResult(false);
              });
          }}
          isLoading={isLoadingResult}
        />
        {result && (
          <>
            <Divider />
            <Prediction predictionResult={result} />
          </>
        )}
      </Stack>
    </Layout>
  );
};

export default Home;
