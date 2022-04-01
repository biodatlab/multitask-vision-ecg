import {
  Box,
  Button,
  Code,
  Container,
  Flex,
  Heading,
  Input,
  Radio,
  RadioGroup,
  Stack,
  Table,
  TableContainer,
  Tbody,
  Td,
  Text,
  Th,
  Thead,
  Tr,
  VStack,
} from "@chakra-ui/react";
import { useState } from "react";
import Bar from "../components/bar";
import Layout from "../components/layout";

interface Answers {
  age?: number;
  sex?: number;
  history_heart_failure?: number;
  history_cad?: number;
  implantable?: number;
  diabetes?: number;
  hypertension?: number;
  smoking?: number;
  renal_replacement?: number;
  lvef?: number;
}

interface Question {
  key:
    | "age"
    | "sex"
    | "history_heart_failure"
    | "history_cad"
    | "implantable"
    | "diabetes"
    | "hypertension"
    | "smoking"
    | "renal_replacement"
    | "lvef";
  text: string;
  type: string;
  optional: boolean;
  choices: Array<string>;
}

// all inputs are number
const questions: Array<Question> = [
  {
    key: "age",
    text: "อายุ (ปี)",
    type: "input",
    optional: false,
    choices: [],
  },
  {
    key: "sex",
    text: "เพศ",
    type: "radio",
    optional: false,
    choices: ["ชาย", "หญิง"],
  },
  {
    key: "history_heart_failure",
    text: "มีประวัติโรคหัวใจล้มเหลว",
    type: "radio",
    optional: false,
    choices: ["ไม่มี", "มี"],
  },
  {
    key: "history_cad",
    text: "มีประวัติโรคกล้ามเนื้อหัวใจขาดเลือด",
    type: "radio",
    optional: false,
    choices: ["ไม่มี", "มี"],
  },
  {
    key: "implantable",
    text: "มีอุปกรณ์ฝังช่วย",
    type: "radio",
    optional: false,
    choices: ["ไม่มี", "มี"],
  },
  {
    key: "diabetes",
    text: "เป็นโรคเบาหวาน",
    type: "radio",
    optional: false,
    choices: ["ไม่ใช่", "ใช่"],
  },
  {
    key: "hypertension",
    text: "เป็นโรคความดันสูง",
    type: "radio",
    optional: false,
    choices: ["ไม่ใช่", "ใช่"],
  },
  {
    key: "smoking",
    text: "เคยสูบบุหรี่",
    type: "radio",
    optional: false,
    choices: ["ไม่ใช่", "ใช่"],
  },
  {
    key: "renal_replacement",
    text: "เคยฟอกไต",
    type: "radio",
    optional: false,
    choices: ["ไม่ใช่", "ใช่"],
  },
  {
    key: "lvef",
    text: "ค่าประสิทธิภาพหัวใจห้องล่างซ้าย (ไม่บังคับ)",
    type: "input",
    optional: true,
    choices: [],
  },
];

const getRiskLabel = (percent: number) => {
  if (percent < 3) {
    return "ความเสี่ยงต่ำ";
  }

  if (percent >= 3 && percent < 5) {
    return "ความเสี่ยงปานกลาง";
  }

  if (percent >= 5 && percent < 10) {
    return "ความเสี่ยงสูง";
  }

  return "ความเสี่ยงสูงมาก";
};

const calculate = (
  answers: Answers,
  onInvalid: (invalidFields: Array<string>) => void
): number | null => {
  const {
    age,
    sex,
    history_heart_failure,
    history_cad,
    implantable,
    diabetes,
    hypertension,
    smoking,
    renal_replacement,
    lvef,
  } = answers;

  if (
    age === undefined ||
    sex === undefined ||
    history_heart_failure === undefined ||
    history_cad === undefined ||
    implantable === undefined ||
    diabetes === undefined ||
    hypertension === undefined ||
    smoking === undefined ||
    renal_replacement === undefined
  ) {
    const requiredFields = questions
      .map((x) => x.key)
      .filter((x) => x !== "lvef");

    const invalidFields = requiredFields.filter((k) => {
      return answers[k] === undefined;
    });

    onInvalid(invalidFields);

    return null;
  }

  let prognostic_index;
  let prediction;

  if (!lvef) {
    prognostic_index =
      0.02 * age +
      0.452 * sex +
      1.019 * history_heart_failure +
      0.579 * history_cad +
      0.453 * implantable +
      0.704 * diabetes +
      0.297 * hypertension +
      0.506 * smoking +
      1.508 * renal_replacement;
    prediction = 1 - 0.99535167 ** Math.exp(prognostic_index);
  } else {
    prognostic_index =
      0.025 * age +
      0.563 * sex +
      0.772 * history_heart_failure +
      0.463 * history_cad +
      0.413 * implantable +
      0.732 * diabetes +
      0.356 * hypertension +
      0.523 * smoking +
      1.516 * renal_replacement -
      0.021 * lvef;
    prediction = 1 - 0.98801461 ** Math.exp(prognostic_index);
  }

  return prediction * 100;
};

const Manual = () => {
  const [answers, setAnswers] = useState({} as Answers);
  const [invalidFields, setInvalidFields] = useState([] as Array<string>);
  const [displayFormula, setDisplayFormula] = useState(false);
  const [prediction, setPrediction] = useState(null as number | null);

  return (
    <Layout>
      <Stack direction="column" gap={2} py={6}>
        <Heading as="h1">
          แบบประเมินความเสี่ยงโรคหัวใจล้มเหลวภายในระยะเวลา 3 ปี
        </Heading>
        <Text>
          กรอกแบบสอบถามเพื่อประเมินความเสี่ยงที่จะเกิดโรคหัวใจล้มเหลวภายในระยะเวลา 3 ปี
          โดยความเสี่ยงแบ่งได้เป็น 4 ระดับขึ้นกับผลการทำนาย (1) ความเสี่ยงต่ำ ผลการทำนายน้อยกว่า 3% (2)
          ความเสี่ยงปานกลาง ผลการทำนายระหว่าง 3 ถึง 5 % (3) ความเสียงสูง
          ผลการทำนายระหว่่าง 5 ถึง 10 % (4) ความเสี่ยงสูงมาก ผลการทำนายมากกว่า 10%
        </Text>
        <VStack gap={4}>
          <TableContainer>
            <Table variant="simple">
              <Thead>
                <Tr>
                  <Th>คำถาม</Th>
                  <Th>คำตอบ</Th>
                </Tr>
              </Thead>
              <Tbody>
                {questions.map((question) => (
                  <Tr
                    key={question.text}
                    background={
                      invalidFields.includes(question.key)
                        ? "red.100"
                        : undefined
                    }
                  >
                    <Td>{question.text}</Td>
                    <Td>
                      {question.type === "radio" ? (
                        <RadioGroup
                          name={question.key}
                          onChange={(val) => {
                            setAnswers((prev) => ({
                              ...prev,
                              [question.key]: val,
                            }));
                            // remove from invalid fields
                            setInvalidFields((prev) =>
                              prev.filter((x) => x !== question.key)
                            );
                          }}
                          value={answers[question.key]}
                        >
                          <Stack direction="row">
                            {question.choices.map((choice, ind) => (
                              <Radio
                                key={choice}
                                value={String(ind)}
                                isRequired={!question.optional}
                                isInvalid={invalidFields.includes(question.key)}
                              >
                                {choice}
                              </Radio>
                            ))}
                          </Stack>
                        </RadioGroup>
                      ) : (
                        <Input
                          size="sm"
                          type="number"
                          value={answers[question.key] || ""}
                          isRequired={!question.optional}
                          isInvalid={invalidFields.includes(question.key)}
                          onChange={(e) => {
                            setAnswers((prev) => ({
                              ...prev,
                              [question.key]: Number(e.target.value),
                            }));
                            // remove from invalid fields
                            setInvalidFields((prev) =>
                              prev.filter((x) => x !== question.key)
                            );
                          }}
                        />
                      )}
                    </Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          </TableContainer>
          {invalidFields.length > 0 && (
            <Box mt={4}>
              <Text fontWeight="bold" color="red.500">
                กรุณาเลือก{" "}
                {invalidFields
                  .map((x) => `"${questions.find((y) => y.key === x)?.text}"`)
                  .join(", ")}
              </Text>
            </Box>
          )}

          <Button
            onClick={() => {
              const pred = calculate(answers, (invalidFields) => {
                setInvalidFields(invalidFields);
              });

              setPrediction(pred);
            }}
            colorScheme="pink"
            px={6}
          >
            คำนวณ
          </Button>

          <Box
            maxW="100%"
            p={4}
            textAlign="center"
            display={prediction ? "block" : "none"}
          >
            <Text fontSize="3xl" fontWeight="bold">
              {`${prediction?.toFixed(0)}%`}
            </Text>
            <Text fontSize="xl">{getRiskLabel(prediction!)}</Text>
            <Bar value={prediction!} min={0} max={12} />
            <Button
              variant="ghost"
              colorScheme="pink"
              onClick={() => setDisplayFormula((prev) => !prev)}
            >
              ดูสูตรคำนวณความเสี่ยง
            </Button>
          </Box>

          <Box
            borderRadius={8}
            background="pink.50"
            p={8}
            pb={6}
            display={displayFormula ? "block" : "none"}
          >
            <Heading as="h3" fontSize="lg">
              เมื่อมีค่าประสิทธิภาพหัวใจห้องล่างซ้าย (LVEF)
            </Heading>
            <Code border="2px solid pink" borderRadius={8} p={4} mb={4}>
              <Stack direction="column" gap={4}>
                <Stack direction="row">
                  <Box w="135px" textAlign="right">
                    prognostic_index
                  </Box>
                  <Box>=</Box>
                  <Box>
                    (0.025 * อายุ) + (0.563 * เพศ) + (0.772 *
                    ประวัติโรคหัวใจล้มเหลว) + (0.463 *
                    ประวัติโรคกล้ามเนื้อหัวใจขาดเลือด) + (0.413 *
                    อุปกรณ์ฝังช่วย) + (0.732 * โรคเบาหวาน) + (0.356 *
                    โรคความดันสูง) + (0.523 * สูบบุหรี่) + (1.516 * เคยฟอกไต) -
                    (0.021 * ค่าประสิทธิภาพหัวใจห้องล่างซ้าย)
                  </Box>
                </Stack>
                <Stack direction="row">
                  <Box w="135px" textAlign="right">
                    ความเสี่ยง
                  </Box>
                  <Box>=</Box>
                  <Box>1 - 0.98801461 ** (exp(prognostic_index))</Box>
                </Stack>
              </Stack>
            </Code>
            <Heading as="h3" fontSize="lg">
              เมื่อไม่มีค่าประสิทธิภาพหัวใจห้องล่างซ้าย (No LVEF)
            </Heading>
            <Code border="2px solid pink" borderRadius={8} p={4} mb={4}>
              <Stack direction="column" gap={4}>
                <Stack direction="row">
                  <Box w="135px" textAlign="right">
                    prognostic_index
                  </Box>
                  <Box>=</Box>
                  <Box>
                    (0.020 * อายุ) + (0.452 * เพศ) + (1.019 *
                    ประวัติโรคหัวใจล้มเหลว) + (0.579*
                    ประวัติโรคกล้ามเนื้อหัวใจขาดเลือด) + (0.453 *
                    อุปกรณ์ฝังช่วย) + (0.704 * โรคเบาหวาน) + (0.297 *
                    โรคความดันสูง) + (0.506 * สูบบุหรี่) + (1.508 * เคยฟอกไต)
                  </Box>
                </Stack>
                <Stack direction="row">
                  <Box w="135px" textAlign="right">
                    ความเสี่ยง
                  </Box>
                  <Box>=</Box>
                  <Box>1 - 0.99535167 ** (exp(prognostic_index))</Box>
                </Stack>
              </Stack>
            </Code>
          </Box>
        </VStack>
      </Stack>
    </Layout>
  );
};

export default Manual;
