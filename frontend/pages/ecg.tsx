import {
  Alert,
  AlertDescription,
  AlertIcon,
  AlertTitle,
  Box,
  Button,
  Container,
  Divider,
  Heading,
  Stack,
  Table,
  TableContainer,
  Tbody,
  Td,
  Text,
  Th,
  Thead,
  Tr,
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

const Ecg: NextPage = () => {
  const [result, setResult] = useState(null as predictionResult);
  const [isLoadingResult, setIsLoadingResult] = useState(false);
  const [displayHowToCreateModel, setDisplayHowToCreateModel] = useState(false);

  return (
    <Layout>
      <Box my={12}>
        {/* main content and dropzone */}
        <Box position="relative" textAlign="center" borderRadius="3xl" py={10}>
          <Box
            position="absolute"
            borderRadius="3xl"
            top={0}
            left={0}
            w="100%"
            h="29.5em"
            backgroundColor="secondary.50"
          />
          <Container maxW="container.sm" position="relative">
            <Heading
              as="h1"
              fontSize={40}
              lineHeight="tall"
              fontWeight="semibold"
              color="secondary.400"
              mb={2}
            >
              AI ประเมินความเสี่ยง
              <br />
              จากภาพสแกนคลื่นไฟฟ้าหัวใจ
            </Heading>

            <Box maxW="container.sm" px={[0, 20]}>
              <Text mb={8}>
                AI
                ทำนายความน่าจะเป็นของการมีรอยแผลเป็นในหัวใจและค่าประสิทธิภาพการทำงานของหัวใจห้องล่างซ้ายจากภาพสแกนคลื่นไฟฟ้าหัวใจ
                (Electrocardiogram, ECG) แบบ 12 Lead
              </Text>

              <Box left={[0, "50%"]} right={[0, "50%"]}>
                <Dropzone
                  onClearFile={() => setResult(null)}
                  onSubmit={(f) => {
                    setIsLoadingResult(true);
                    console.log(f);

                    const formdata = new FormData();
                    formdata.append("file", f, f.name);

                    fetch("/api/predict", {
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
              </Box>
            </Box>
          </Container>
        </Box>

        {/* if prediction result valid */}
        <Box display={result && Array.isArray(result) ? undefined : "none"}>
          <Prediction
            onClickHowTo={() => setDisplayHowToCreateModel((prev) => !prev)}
            predictionResult={result}
          />
        </Box>

        {/* if prediction result error */}
        <Box display={result && !Array.isArray(result) ? undefined : "none"}>
          <Alert
            status="error"
            variant="subtle"
            colorScheme="gray"
            flexDirection="column"
            alignItems="center"
            justifyContent="center"
            height="200px"
            textAlign="center"
            borderRadius="3xl"
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
        </Box>
      </Box>
      {/* <Stack py={6} direction="column" textAlign={"center"} gap={2}>
        <Heading color="gray.500" as={"h1"}>
          หทัย AI
        </Heading>
        <Text textAlign="left">
          <Text as="span" fontWeight="bold">
            หทัย AI
          </Text>{" "}
          เป็นแอพพลิเคชั่นที่ใช้ AI ในการทำนายผลของภาพสแกนคลื่นไฟฟ้าหัวใจ
          (Electrocardiogram, ECG) แบบ 12 Lead ในรูปแบบของภาพหรือไฟล์ PDF
          โดยโมเดล{" "}
          <Button
            colorScheme="pink"
            variant="link"
            onClick={() => setDisplayHowToCreateModel((prev) => !prev)}
          >
            (ดูวิธีการสร้างโมเดล)
          </Button>{" "}
          จะทำนายความน่าจะเป็นของการมีรอยแผลเป็นในหัวใจ (Myocardial scar)
          และค่าประสิทธิภาพการทำงานของหัวใจห้องล่างซ้าย (Left ventricular
          ejection fraction, LVEF) นอกจากนั้นผู้ใช้งานยังสามารถทำแบบสอบถาม
          เพื่อประเมินความเสี่ยงเกี่ยวกับโรคหัวใจผ่านแอพพลิเคชั่นได้อีกด้วย
        </Text>
        <Divider />
        <Heading as="h2" size="md">
          ประเมินความเสี่ยงจากภาพสแกนคลื่นไฟฟ้าหัวใจแบบ 12 ลีด
        </Heading>
        <Text textAlign="center">
          อัพโหลดไฟล์สแกนคลื่นไฟฟ้าหัวใจแบบ 12 ลีดในกล่องด้านล่างเพื่อทำนายผล
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
            <Prediction
              onClickHowTo={() => setDisplayHowToCreateModel((prev) => !prev)}
              predictionResult={result}
            />
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
        {displayHowToCreateModel && (
          <>
            <Divider />
            <Stack direction="column" gap={2} textAlign="left">
              <Heading as="h3" size="md">
                วิธีการสร้างและวัดผลความแม่นยำของโมเดล
              </Heading>
              <Text>
                โมเดลทำนายผลของภาพสแกนคลื่นไฟฟ้าหัวใจ (Electrocardiogram, ECG)
                แบบ 12 Lead
                ที่ใช้ทำนายนี้ถูกสร้างขึ้นมาจากฐานข้อมูลภาพสแกนคลื่นไฟฟ้าหัวใจจำนวน
                2100 ภาพที่เก็บจากศูนย์โรคหัวใจโรงพยาบาลศิริราช
                โดยเป็นข้อมูลของคลื่นไฟฟ้าหัวใจที่ไม่มีแผลเป็น 1100
                ภาพและมีแผลเป็น 1000 ภาพ
              </Text>
              <Text>
                ในการสร้างโมเดลการทำนาย
                ข้อมูลถูกแบ่งเป็นชุดข้อมูลสำหรับสร้างโมเดล,
                วัดผลระหว่างคำนวณโมเดล, และทดสอบโมเดล ด้วยสัดส่วน 80:10:10
                ทั้งนี้โมเดลอาจมีความผิดพลาดในการทำนายผลได้
                ผู้ใช้งานสามารถดูความถูกต้องเบื้องต้นได้ตามตารางด้านล่าง (ผลเป็น
                %)
              </Text>
              <Heading as="h3" size="md">
                การวัดผลความแม่นยำของโมเดล
              </Heading>
              <TableContainer>
                <Table variant="simple">
                  <Thead>
                    <Tr>
                      <Th>โมเดล</Th>
                      <Th isNumeric>ความแม่นยำ (Accuracy)</Th>
                      <Th isNumeric>ความจำเพาะ (Specificity)</Th>
                      <Th isNumeric>ความไว (Sensitivity)</Th>
                      <Th isNumeric>AUC</Th>
                    </Tr>
                  </Thead>
                  <Tbody>
                    <Tr>
                      <Td>แผลเป็นในหัวใจ (Mycardial scar)</Td>
                      <Td isNumeric>80.5</Td>
                      <Td isNumeric>85.0</Td>
                      <Td isNumeric>80.5</Td>
                      <Td isNumeric>89.1</Td>
                    </Tr>
                    <Tr>
                      <Td>LVEF &lt; 40</Td>
                      <Td isNumeric>89.0</Td>
                      <Td isNumeric>88.6</Td>
                      <Td isNumeric>89.0</Td>
                      <Td isNumeric>92.9</Td>
                    </Tr>
                    <Tr>
                      <Td>LVEF &lt; 50</Td>
                      <Td isNumeric>84.8</Td>
                      <Td isNumeric>84.2</Td>
                      <Td isNumeric>84.8</Td>
                      <Td isNumeric>90.5</Td>
                    </Tr>
                  </Tbody>
                </Table>
              </TableContainer>
            </Stack>
          </>
        )}
      </Stack> */}
    </Layout>
  );
};

export default Ecg;
