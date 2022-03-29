import {
  Alert,
  AlertDescription,
  AlertIcon,
  AlertTitle,
  Divider,
  Heading,
  Stack,
  Text,
} from "@chakra-ui/react";
import type { NextPage } from "next";
import { useState } from "react";
import Dropzone from "../components/dropzone";
import Layout from "../components/layout";
import Prediction from "../components/prediction";

interface prediction {
  prediction_title: string;
  score: number;
  labelLt: string;
  labelRt: string;
}

export type predictionResult = Array<prediction> | { error: string } | null;

const Home: NextPage = () => {
  const [result, setResult] = useState(null as predictionResult);
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
        {Array.isArray(result) && (
          <>
            <Divider />
            <Prediction predictionResult={result} />
          </>
        )}
        {result && !Array.isArray(result) && (
          <>
            <Divider />
            <Alert
              status="error"
              variant="subtle"
              flexDirection="column"
              alignItems="center"
              justifyContent="center"
              textAlign="center"
              height="200px"
            >
              <AlertIcon boxSize="40px" mr={0} />
              <AlertTitle mt={4} mb={1} fontSize="lg">
                ไม่สามารถทำนายได้
              </AlertTitle>
              <AlertDescription maxWidth="sm">
                อาจมีข้อผิดพลาดเกี่ยวกับรูปภาพหรือไฟล์ที่ส่งมา
                กรุณาลองใหม่อีกครั้งหรือติดต่อเรา
              </AlertDescription>
            </Alert>
          </>
        )}
      </Stack>
    </Layout>
  );
};

export default Home;
