import {
  Box,
  Flex,
  Grid,
  GridItem,
  Heading,
  Spacer,
  Stack,
  Text,
} from "@chakra-ui/react";
import { prediction, predictionResult } from "../../pages/ecg";

interface PredictionProps {
  predictionResult: predictionResult;
}

interface PredictionCardProps {
  data: prediction;
}

interface ProbabilityBarProps {
  probability: number;
  average?: number;
}

const ProbabilityBar = ({ probability, average }: ProbabilityBarProps) => {
  const height = 24;
  const shift = height / 2 + 4;

  return (
    <Flex maxW="100%" h={height} flex={1} position="relative">
      {/* top labels */}
      <Flex
        width="100%"
        justifyContent="space-between"
        position="absolute"
        top={shift - 3 - 10 + 2}
      >
        <Text as="span" color="gray.800" fontSize="xs">
          ความเสี่ยงต่ำ
        </Text>
        <Text as="span" color="gray.800" fontSize="xs">
          ความเสี่ยงสูง
        </Text>
      </Flex>
      {/* bg bar */}
      <Box
        h={8}
        w="100%"
        position="absolute"
        top={shift - 4}
        bgGradient="linear(to-r, secondary.400, primary.300)"
      />
      {/* overlay gray bar */}
      <Box
        h={8}
        w={`${100 - probability}%`}
        position="absolute"
        top={shift - 4}
        right={0}
        backgroundColor="gray.100"
      />
      {/* inner percent text */}
      <Box
        position="absolute"
        top="53px"
        right={probability > 15 ? `${102 - probability}%` : undefined}
        left={probability > 15 ? undefined : `${2 + probability}%`}
        zIndex={2}
      >
        <Text
          as="span"
          color={probability > 15 ? "white" : "black"}
          fontSize="md"
          fontWeight="semibold"
        >
          {`${Math.round(probability)}%`}
        </Text>
      </Box>
      {average && (
        <>
          {/* average pole */}
          <Box
            h={10}
            w="2px"
            position="absolute"
            top={shift - 5 - 1}
            left={`calc(${average}% - 1px)`}
            backgroundColor="gray.800"
          />
          {/* average label */}
          <Box
            position="absolute"
            top={shift - 3 - 10 + 2}
            left={`calc(${average}% - 18px)`}
          >
            <Text as="span" color="gray.900" fontSize={10}>
              ค่าเฉลี่ย
            </Text>
          </Box>
        </>
      )}
      {/* bottom labels */}
      <Flex
        width="100%"
        justifyContent="space-between"
        position="absolute"
        bottom={0 - 2}
      >
        <Text as="span" color="gray.700" fontSize="xs">
          0
        </Text>
        <Text as="span" color="gray.700" fontSize="xs">
          25
        </Text>
        <Text as="span" color="gray.700" fontSize="xs">
          50
        </Text>
        <Text as="span" color="gray.700" fontSize="xs">
          75
        </Text>
        <Text as="span" color="gray.700" fontSize="xs">
          100%
        </Text>
      </Flex>
    </Flex>
  );
};

export const PredictionCard = ({ data }: PredictionCardProps) => {
  const { title, description, risk_level, probability, average } = data;

  return (
    <Flex
      flex={1}
      w="100%"
      h="100%"
      maxW={{ base: "lg", lg: "100%" }}
      flexDirection="column"
      alignItems="center"
      borderRadius="2xl"
      boxShadow="lg"
      p={6}
      mx="auto"
    >
      <Flex w="100%" justifyContent="space-between" alignItems="center" mb={4}>
        <Box textAlign="left">
          <Heading as="h5" fontSize="2xl" color="secondary.400" mb={1}>
            {title}
          </Heading>
          <Text fontSize="sm" maxW="2xs" pr={2}>
            {description}
          </Text>
        </Box>
        <Box
          textAlign="center"
          backgroundColor={
            risk_level === "ปานกลาง"
              ? "gray.50"
              : risk_level === "สูง"
              ? "primary.50"
              : "secondary.50"
          }
          borderRadius="lg"
          p={3}
        >
          <Text fontSize="sm" mb={3}>
            ระดับความเสี่ยง
          </Text>
          <Heading
            as="h5"
            fontSize="xl"
            color={
              risk_level === "ปานกลาง"
                ? "gray.600"
                : risk_level === "สูง"
                ? "primary.300"
                : "secondary.400"
            }
            mb={1}
          >
            {risk_level}
          </Heading>
        </Box>
      </Flex>

      <Spacer />

      <Flex w="100%" justifyContent="center">
        <ProbabilityBar probability={probability} average={average} />
      </Flex>
    </Flex>
  );
};

const Prediction = ({ predictionResult }: PredictionProps) => {
  if (!Array.isArray(predictionResult)) {
    return null;
  }

  return (
    <Stack direction="column" gap={4} mt={10}>
      <Heading as="h4" fontSize="2xl" color="secondary.400" mb={6}>
        ผลทำนาย
      </Heading>
      <Grid
        templateColumns={{ base: "repeat(1, 1fr)", lg: "repeat(2, 1fr)" }}
        gap={6}
      >
        {predictionResult.map((data) => (
          <GridItem key={data.title}>
            <PredictionCard data={data} />
          </GridItem>
        ))}
      </Grid>
    </Stack>
  );
};

export default Prediction;
