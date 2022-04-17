import {
  Box,
  Button,
  FormControl,
  FormLabel,
  forwardRef,
  HStack,
  Input,
  Radio,
  RadioGroup,
  RadioProps,
  Stack,
  StackDivider,
  VStack,
} from "@chakra-ui/react";
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

  return (
    <Layout>
      <Stack direction="column" gap={2} py={6}></Stack>
      <RadioGroup name={"test"}>
        <Stack direction="row">
          <form onSubmit={handleSubmit(onSubmit)}>
            <VStack gap={6} divider={<StackDivider borderColor="gray.200" />}>
              {Object.entries(assessmentQuestions).map(([key, value]) => {
                if (value.type === "input") {
                  return (
                    <FormControl key={key} isRequired={!value.optional}>
                      <FormLabel
                        htmlFor={value.key}
                      >{`${key}. ${value.text}`}</FormLabel>
                      <Box pl={5}>
                        <Input
                          id={value.key}
                          type="number"
                          w={24}
                          {...register(value.key)}
                        />
                      </Box>
                      {/* <FormHelperText>We'll never share your email.</FormHelperText> */}
                    </FormControl>
                  );
                }

                if (value.type === "radio") {
                  return (
                    <FormControl key={key} isRequired={!value.optional}>
                      <FormLabel
                        htmlFor={value.key}
                      >{`${key}. ${value.text}`}</FormLabel>
                      <RadioGroup pl={5}>
                        <HStack>
                          {value.choices.map((choice, ind) => {
                            return (
                              <WrappedRadio
                                key={choice}
                                customProps={{
                                  checked: watchAll[value.key] === String(ind),
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
            <Button
              mt={4}
              colorScheme="primary"
              backgroundColor="primary.300"
              isLoading={isSubmitting}
              type="submit"
            >
              คำนวณความเสี่ยง
            </Button>
          </form>
        </Stack>
      </RadioGroup>
    </Layout>
  );
};

export default Assessment;
