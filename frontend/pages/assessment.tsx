import {
  Box,
  Button,
  Checkbox,
  CheckboxGroup,
  Container,
  Divider,
  Flex,
  FormControl,
  FormLabel,
  forwardRef,
  Heading,
  HStack,
  Input,
  Radio,
  RadioGroup,
  RadioProps,
  StackDivider,
  Text,
  VStack,
} from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import Layout from "../components/layout";
import assessmentQuestions from "../data/assessmentQuestions.json";

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

const WrappedRadio = forwardRef<
  { radioProps: RadioProps; customProps: { checked: boolean } },
  "div"
>(({ radioProps, customProps }, ref) => (
  <Box
    w={24}
    backgroundColor={customProps.checked ? "secondary.50" : undefined}
    borderRadius="xl"
    borderWidth="1px"
    borderStyle="solid"
    borderColor="gray.200"
    p={2}
    pl={3}
  >
    <Radio
      colorScheme="secondary"
      sx={{
        "&[aria-checked=true], &[data-checked]": {
          background: "secondary.400",
          borderColor: "secondary.400",
        },
      }}
      ref={ref}
      {...radioProps}
    />
  </Box>
));

const questionsMap: { [index: string]: number[] } = {
  scar: [2, 4, 7, 8, 9, 10, 11],
  cadScar: [2, 5, 7, 8, 9, 11],
  lvef40: [1, 2, 3, 4, 8, 9, 10, 11],
  lvef50: [1, 2, 3, 4, 6, 8, 9, 10],
};

const allQuestions: { [index: string]: any } = assessmentQuestions;
const defaultQuestions = Object.entries(assessmentQuestions).map(
  ([, value]) => value
);

const Assessment = () => {
  const {
    register,
    handleSubmit,
    watch,
    formState: { errors, isSubmitting },
  } = useForm();
  const onSubmit = (data: any) => {
    console.log(data);
  };

  // use watch to help style radio buttons
  const watchAll = watch();
  const watchDiseaseSelection = watch("diseases_selection");

  type InferredQuestionType = Omit<
    typeof assessmentQuestions["1"],
    "choices"
  > & {
    choices: never[] | string[];
  };

  // Object.entries(assessmentQuestions).map(([key, value]) => value)

  useEffect(() => {
    const diseases_selection = watchDiseaseSelection;

    if (Array.isArray(diseases_selection)) {
      const questionsSet = new Set<number>(
        diseases_selection.reduce(
          (acc, cur) => [...acc, ...questionsMap[cur]],
          []
        )
      );
      const questionKeys = Array.from(questionsSet).sort((a, b) => a - b);
      const questions = questionKeys.map((key) => allQuestions[String(key)]);

      setRequiredQuestions(questions);
    }

    // console.log({ watchAll });
  }, [watchDiseaseSelection]);

  const [requiredQuestions, setRequiredQuestions] =
    useState<InferredQuestionType[]>(defaultQuestions);

  return (
    <Layout>
      <Box my={12}>
        {/* main content */}
        <Box
          position="relative"
          textAlign="center"
          borderRadius="3xl"
          pt={10}
          pb={12}
          mb={8}
        >
          <Box
            position="absolute"
            borderRadius="3xl"
            top={0}
            left={0}
            w="100%"
            h="100%"
            backgroundColor="white"
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
              แบบประเมินความเสี่ยง
              <br />
              ภาวะโรคหัวใจต่าง ๆ
            </Heading>

            <Box maxW="container.sm" px={[0, 25]}>
              <Text>
                แบบประเมินความเสี่ยงของการมีรอยแผลเป็นในหัวใจ (Myocardial Scar)
                โรคหลอดเลือดแดงของหัวใจตีบหรือตัน (Coronary Artery Disease, CAD)
                และการบีบตัวของหัวใจห้องล่างซ้ายผิดปกติ (LVEF)
                สำหรับประชาชนทั่วไปหรือผู้ที่มีความเสี่ยง
              </Text>
            </Box>
          </Container>
        </Box>

        {/* questions form */}
        <Box
          maxW="2xl"
          backgroundColor="white"
          borderRadius="2xl"
          boxShadow="lg"
          py={10}
          px={14}
          mx="auto"
        >
          <form onSubmit={handleSubmit(onSubmit)}>
            {/* disease selection */}
            <FormControl mb={6}>
              <FormLabel>
                <Heading as="h6" fontSize="md" color="secondary.400" mb={6}>
                  กรุณาเลือกภาวะโรคที่ต้องการประเมินความเสี่ยง
                </Heading>
              </FormLabel>
              <CheckboxGroup
                colorScheme="secondary"
                defaultValue={["scar", "cadScar", "lvef40", "lvef50"]}
              >
                <VStack
                  alignItems="flex-start"
                  sx={{
                    "& .chakra-checkbox__control[data-checked]": {
                      backgroundColor: "secondary.400",
                      borderColor: "secondary.400",
                    },
                  }}
                >
                  <Checkbox value="scar" {...register("diseases_selection")}>
                    รอยแผลเป็นในหัวใจ
                  </Checkbox>
                  <Checkbox value="cadScar" {...register("diseases_selection")}>
                    หลอดเลือดแดงของหัวใจตีบหรือตัน
                  </Checkbox>
                  <Checkbox value="lvef40" {...register("diseases_selection")}>
                    ประสิทธิภาพการบีบตัวของหัวใจห้องล่างซ้ายต่ำกว่า 40%
                  </Checkbox>
                  <Checkbox value="lvef50" {...register("diseases_selection")}>
                    ประสิทธิภาพการบีบตัวของหัวใจห้องล่างซ้ายต่ำกว่า 50%
                  </Checkbox>
                </VStack>
              </CheckboxGroup>
            </FormControl>

            {/* question to acquire */}
            {requiredQuestions?.length > 0 && (
              <Box>
                <Heading as="h6" fontSize="md" color="secondary.400" mb={6}>
                  กรุณาตอบคำถามด้านล่างให้ครบ
                  และกดปุ่มคำนวณความเสี่ยงด้านล่างเพื่อรับผลการประเมินความเสี่ยง
                </Heading>

                <VStack
                  gap={6}
                  divider={<StackDivider borderColor="gray.200" />}
                >
                  {requiredQuestions.map((value, ind) => {
                    if (value.type === "input") {
                      return (
                        <FormControl
                          key={value.key}
                          isRequired={!value.optional}
                        >
                          <FormLabel htmlFor={value.key}>{`${ind + 1}. ${
                            value.text
                          }`}</FormLabel>
                          <Box pl={5}>
                            <Input
                              id={value.key}
                              type="number"
                              w={24}
                              {...register(value.key)}
                            />
                          </Box>
                        </FormControl>
                      );
                    }

                    if (value.type === "radio") {
                      return (
                        <FormControl
                          key={value.key}
                          isRequired={!value.optional}
                        >
                          <FormLabel htmlFor={value.key}>{`${ind + 1}. ${
                            value.text
                          }`}</FormLabel>
                          <RadioGroup pl={5}>
                            <HStack>
                              {value.choices.map((choice, ind) => {
                                return (
                                  <WrappedRadio
                                    key={choice}
                                    customProps={{
                                      checked:
                                        watchAll[value.key] === String(ind),
                                    }}
                                    radioProps={{
                                      children: choice,
                                      value: String(ind),
                                      ...register(value.key),
                                    }}
                                  />
                                );
                              })}
                            </HStack>
                          </RadioGroup>
                          {/* <FormHelperText>We'll never share your email.</FormHelperText> */}
                        </FormControl>
                      );
                    }
                  })}
                </VStack>
                <Divider borderColor="gray.200" mt={8} />
                <Flex justifyContent="flex-end" pt={2}>
                  <Button
                    mt={4}
                    colorScheme="primary"
                    backgroundColor="primary.300"
                    isLoading={isSubmitting}
                    type="submit"
                  >
                    คำนวณความเสี่ยง
                  </Button>
                </Flex>
              </Box>
            )}
          </form>

          {/* hacky back faint secondary bg */}
          <Box
            zIndex={-1}
            position={"fixed"}
            top={0}
            left={0}
            width="100vw"
            height="100vh"
            background="secondary.50"
          />
        </Box>
      </Box>
    </Layout>
  );
};

export default Assessment;
